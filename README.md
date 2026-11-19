# SM80 packed GDN decode benchmark

Run the kernel reproduction on one A100-SXM4-40GB with an existing vLLM 0.29 CUDA environment.

```sh
cd kernel
python3 bench.py --out trial-01
```

See kernel/README.md for the pinned source and defaults. main-fp32-only.jsonl contains all 236 shapes, five paired timings per shape, and correctness checks. abs_max has candidate-versus-baseline output and state differences, then candidate-versus-PyTorch output and state differences.

Serving used the FP32-only runtime patch with GDN_TUNE=0 and 1 and 512 input/output tokens. The commands in serving/cli-and-results.json are argument arrays. Replace MODEL_WEIGHTS, BIND_ADDRESS, and BENCHMARK_ENDPOINT with your own checkpoint, bind address, and endpoint. Configure PYTHONPATH to include serving/runtime-patch, create the relative results directories, and put the GSM8K train/test Parquet files beside your working directory or adjust the local task YAML. The exact local checkpoint files and dataset are not distributed here. Per-run numerical metrics and per-item score flags retain their original values. The per-item score file also contains a separate historical table-based comparison.
