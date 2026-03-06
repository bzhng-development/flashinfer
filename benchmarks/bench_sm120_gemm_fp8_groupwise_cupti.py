"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import csv
import itertools
import math
from importlib.metadata import version as importlib_metadata_version
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from cupti import cupti  # noqa: F401

from flashinfer.gemm import gemm_fp8_nt_blockscaled, gemm_fp8_nt_groupwise
from flashinfer.testing.utils import bench_gpu_time, dequantize_fp8, quantize_fp8
from flashinfer.utils import get_compute_capability, round_up


def _parse_int_csv(v: str) -> List[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def _parse_triplet(v: str) -> Tuple[int, int, int]:
    vals = _parse_int_csv(v)
    if len(vals) != 3:
        raise ValueError(
            f"Expected 3 comma-separated integers for scale granularity, got: {v}"
        )
    return vals[0], vals[1], vals[2]


def _dtype_from_str(v: str) -> torch.dtype:
    vv = v.lower()
    if vv in ("bf16", "bfloat16"):
        return torch.bfloat16
    if vv in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported out dtype: {v}")


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _check_cupti(strict: bool) -> None:
    # cupti-python >= 13 is required for FlashInfer CUPTI timing.
    cupti_version = importlib_metadata_version("cupti-python")
    major = int(cupti_version.split(".")[0])
    if major < 13:
        if strict:
            raise RuntimeError(f"cupti-python>={13} is required, got {cupti_version}")
        print(f"[WARN] cupti-python>={13} is recommended, got {cupti_version}")


def _quantize_with_padding(
    x: torch.Tensor,
    gran_m: int,
    gran_k: int,
    scale_major_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    m_pad = round_up(m, gran_m)
    k_pad = round_up(k, gran_k)

    x_pad = torch.zeros((m_pad, k_pad), dtype=x.dtype, device=x.device)
    x_pad[:m, :k] = x

    if scale_major_mode == "K":
        scale_shape = (m_pad // gran_m, k_pad // gran_k)
    else:
        scale_shape = (k_pad // gran_k, m_pad // gran_m)

    x_fp8_pad, x_scale = quantize_fp8(
        x_pad, scale_shape, (gran_m, gran_k), scale_major_mode
    )
    return x_fp8_pad[:m, :k].contiguous(), x_scale


def _dequantize_with_padding(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    gran_m: int,
    gran_k: int,
    scale_major_mode: str,
) -> torch.Tensor:
    m, k = x_fp8.shape
    if scale_major_mode == "K":
        m_pad = x_scale.shape[0] * gran_m
        k_pad = x_scale.shape[1] * gran_k
    else:
        m_pad = x_scale.shape[1] * gran_m
        k_pad = x_scale.shape[0] * gran_k

    x_fp8_pad = torch.zeros((m_pad, k_pad), dtype=x_fp8.dtype, device=x_fp8.device)
    x_fp8_pad[:m, :k] = x_fp8
    x_deq_pad = dequantize_fp8(x_fp8_pad, x_scale, scale_major_mode)
    return x_deq_pad[:m, :k]


def _compute_io_bytes(
    m: int,
    n: int,
    k: int,
    gran_m: int,
    gran_n: int,
    gran_k: int,
    out_dtype: torch.dtype,
) -> int:
    # FP8 elements are 1 byte each.
    a_bytes = m * k
    b_bytes = n * k
    a_sf_bytes = _ceil_div(m, gran_m) * _ceil_div(k, gran_k) * 4
    b_sf_bytes = _ceil_div(n, gran_n) * _ceil_div(k, gran_k) * 4
    out_bytes = m * n * torch.empty((), dtype=out_dtype).element_size()
    return a_bytes + b_bytes + a_sf_bytes + b_sf_bytes + out_bytes


def _benchmark_case(
    m: int,
    n: int,
    k: int,
    api: str,
    backend: str,
    scale_major_mode: str,
    scale_granularity_mnk: Tuple[int, int, int],
    out_dtype: torch.dtype,
    dry_run_iters: int,
    num_iters: int,
    use_cuda_graph: bool,
    sleep_after_run: bool,
    cold_l2_cache: bool,
    refcheck: bool,
    device: torch.device,
    seed: int,
) -> Dict[str, float]:
    gran_m, gran_n, gran_k = scale_granularity_mnk
    if gran_m <= 0 or gran_n <= 0 or gran_k <= 0:
        raise ValueError(f"Invalid scale granularity: {scale_granularity_mnk}")

    if scale_major_mode not in ("MN", "K"):
        raise ValueError(f"Invalid scale_major_mode: {scale_major_mode}")

    torch.manual_seed(seed)
    a_f32 = torch.randn((m, k), dtype=torch.float32, device=device)
    b_f32 = torch.randn((n, k), dtype=torch.float32, device=device) / math.sqrt(k)

    a_fp8, a_scale = _quantize_with_padding(a_f32, gran_m, gran_k, scale_major_mode)
    b_fp8, b_scale = _quantize_with_padding(b_f32, gran_n, gran_k, scale_major_mode)

    out = torch.empty((m, n), dtype=out_dtype, device=device)

    def _run() -> torch.Tensor:
        if api == "blockscaled":
            return gemm_fp8_nt_blockscaled(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                scale_major_mode=scale_major_mode,
                out=out,
                out_dtype=out_dtype,
            )
        return gemm_fp8_nt_groupwise(
            a=a_fp8,
            b=b_fp8,
            a_scale=a_scale,
            b_scale=b_scale,
            scale_major_mode=scale_major_mode,
            scale_granularity_mnk=scale_granularity_mnk,
            out=out,
            out_dtype=out_dtype,
            backend=backend,
        )

    # Pre-run to trigger JIT/build/caches before measured region.
    _run()
    torch.cuda.synchronize(device)

    if refcheck:
        a_deq = _dequantize_with_padding(
            a_fp8, a_scale, gran_m, gran_k, scale_major_mode
        )
        b_deq = _dequantize_with_padding(
            b_fp8, b_scale, gran_n, gran_k, scale_major_mode
        )
        ref_out = (a_deq @ b_deq.transpose(0, 1)).to(out_dtype)
        torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    times_ms = bench_gpu_time(
        fn=_run,
        dry_run_iters=dry_run_iters,
        repeat_iters=num_iters,
        sleep_after_run=sleep_after_run,
        enable_cupti=True,
        use_cuda_graph=use_cuda_graph,
        cold_l2_cache=cold_l2_cache,
    )
    times = np.asarray(times_ms, dtype=np.float64)

    median_ms = float(np.median(times))
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    p90_ms = float(np.percentile(times, 90))
    p99_ms = float(np.percentile(times, 99))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))

    flops = 2.0 * m * n * k
    tflops = flops / (median_ms * 1e9) if median_ms > 0 else float("inf")

    bytes_total = _compute_io_bytes(m, n, k, gran_m, gran_n, gran_k, out_dtype)
    tbps = bytes_total / (median_ms * 1e9) if median_ms > 0 else float("inf")

    return {
        "m": m,
        "n": n,
        "k": k,
        "api": api,
        "backend": backend,
        "scale_major_mode": scale_major_mode,
        "scale_granularity_m": gran_m,
        "scale_granularity_n": gran_n,
        "scale_granularity_k": gran_k,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "p90_ms": p90_ms,
        "p99_ms": p99_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "tflops": tflops,
        "tbps": tbps,
        "iters": int(times.size),
    }


def _print_case_row(result: Dict[str, float]) -> None:
    print(
        " | ".join(
            [
                f"m={result['m']:>5}",
                f"n={result['n']:>5}",
                f"k={result['k']:>5}",
                f"api={result['api']}",
                f"sg=({result['scale_granularity_m']},"
                f"{result['scale_granularity_n']},{result['scale_granularity_k']})",
                f"median={result['median_ms']:.4f} ms",
                f"p90={result['p90_ms']:.4f} ms",
                f"p99={result['p99_ms']:.4f} ms",
                f"std={result['std_ms']:.4f} ms",
                f"TFLOPS={result['tflops']:.3f}",
                f"TB/s={result['tbps']:.3f}",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CUPTI benchmark for FlashInfer FP8 groupwise/blockscaled GEMM on SM120/SM121."
        )
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["groupwise", "blockscaled"],
        default="groupwise",
        help=(
            "Benchmark API. `groupwise` calls gemm_fp8_nt_groupwise. "
            "`blockscaled` calls gemm_fp8_nt_blockscaled."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cutlass", "trtllm"],
        default="cutlass",
        help="Backend for groupwise API (ignored for blockscaled API).",
    )
    parser.add_argument(
        "--m-list",
        type=str,
        default="4,8,16,32,64,128",
        help="Comma-separated M values to benchmark.",
    )
    parser.add_argument(
        "--n-list",
        type=str,
        default="4096",
        help="Comma-separated N values to benchmark.",
    )
    parser.add_argument(
        "--k-list",
        type=str,
        default="4096",
        help="Comma-separated K values to benchmark.",
    )
    parser.add_argument(
        "--scale-major-mode",
        type=str,
        choices=["MN", "K"],
        default="MN",
        help="Scale layout mode.",
    )
    parser.add_argument(
        "--scale-granularity-mnk",
        type=str,
        default="128,128,128",
        help=(
            "Scale granularity tuple for groupwise API, e.g. 1,128,128 or 128,128,128. "
            "For blockscaled API this is ignored and forced to 128,128,128."
        ),
    )
    parser.add_argument(
        "--out-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16"],
        help="Output dtype.",
    )
    parser.add_argument(
        "--dry-run-iters",
        type=int,
        default=20,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Capture and replay with CUDA Graph for benchmarked launches.",
    )
    parser.add_argument(
        "--sleep-after-run",
        action="store_true",
        help="Sleep briefly after each iteration to reduce thermal throttling.",
    )
    parser.add_argument(
        "--disable-cold-l2-cache",
        action="store_true",
        help="Disable cold-L2 behavior between benchmark iterations.",
    )
    parser.add_argument(
        "--refcheck",
        action="store_true",
        help="Run correctness check (dequantized reference matmul) before timing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device string (e.g. cuda, cuda:0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default=None,
        help="Optional CSV output path for results.",
    )
    parser.add_argument(
        "--strict-cupti",
        action="store_true",
        default=True,
        help=("Require cupti-python>=13 and fail otherwise. Default: True."),
    )
    parser.add_argument(
        "--allow-non-sm120",
        action="store_true",
        help="Allow running on non-SM120/SM121 GPUs.",
    )
    args = parser.parse_args()

    _check_cupti(strict=args.strict_cupti)

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if device.type != "cuda":
        raise RuntimeError(f"Device must be CUDA, got: {device}")

    cc = get_compute_capability(device)
    if cc[0] != 12 and not args.allow_non_sm120:
        raise RuntimeError(
            f"This benchmark targets SM120/SM121 paths. "
            f"Detected compute capability: {cc[0]}.{cc[1]}."
        )

    out_dtype = _dtype_from_str(args.out_dtype)
    m_list = _parse_int_csv(args.m_list)
    n_list = _parse_int_csv(args.n_list)
    k_list = _parse_int_csv(args.k_list)

    if args.api == "blockscaled":
        scale_granularity_mnk = (128, 128, 128)
    else:
        scale_granularity_mnk = _parse_triplet(args.scale_granularity_mnk)

    if args.backend == "trtllm" and args.api == "groupwise":
        # trtllm backend constraints in gemm_fp8_nt_groupwise.
        if scale_granularity_mnk != (1, 128, 128):
            raise ValueError(
                "trtllm backend only supports scale_granularity_mnk=(1,128,128)."
            )
        if args.scale_major_mode != "MN":
            raise ValueError("trtllm backend only supports scale_major_mode=MN.")

    print(
        f"[INFO] device={device}, compute_capability={cc[0]}.{cc[1]}, "
        f"api={args.api}, backend={args.backend}, "
        f"scale_major_mode={args.scale_major_mode}, "
        f"scale_granularity_mnk={scale_granularity_mnk}, "
        f"out_dtype={out_dtype}"
    )
    print(
        f"[INFO] timing=CUPTI, dry_run_iters={args.dry_run_iters}, "
        f"num_iters={args.num_iters}, use_cuda_graph={args.use_cuda_graph}, "
        f"cold_l2_cache={not args.disable_cold_l2_cache}"
    )

    results: List[Dict[str, float]] = []
    for m, n, k in itertools.product(m_list, n_list, k_list):
        result = _benchmark_case(
            m=m,
            n=n,
            k=k,
            api=args.api,
            backend=args.backend,
            scale_major_mode=args.scale_major_mode,
            scale_granularity_mnk=scale_granularity_mnk,
            out_dtype=out_dtype,
            dry_run_iters=args.dry_run_iters,
            num_iters=args.num_iters,
            use_cuda_graph=args.use_cuda_graph,
            sleep_after_run=args.sleep_after_run,
            cold_l2_cache=not args.disable_cold_l2_cache,
            refcheck=args.refcheck,
            device=device,
            seed=args.seed,
        )
        results.append(result)
        _print_case_row(result)

    if args.csv_output is not None:
        out_path = Path(args.csv_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "m",
            "n",
            "k",
            "api",
            "backend",
            "scale_major_mode",
            "scale_granularity_m",
            "scale_granularity_n",
            "scale_granularity_k",
            "median_ms",
            "mean_ms",
            "std_ms",
            "p90_ms",
            "p99_ms",
            "min_ms",
            "max_ms",
            "tflops",
            "tbps",
            "iters",
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
