# Reproduce the A100 packed GDN experiment

Use a single A100-SXM4-40GB with a working vLLM 0.29 CUDA environment. The paired module is derived from vLLM commit 578be86c94a26dda09eceb00c67c7ca62deefeb0. It contains the pinned source and a candidate wrapper bound to the same kernel, with the relative helper import resolved from the installed vLLM module. The patch file shows the two upstream files changed.

Run from this directory on an allocated GPU. Set CUDA_VISIBLE_DEVICES to that card in the host environment, or expose only that device to the container. The run validates that exactly one SM80 device is visible.

```sh
python3 bench.py --out trial-01
```

The script refuses to overwrite an existing output directory. It appends one JSON record per finished shape to trial-01/rows.jsonl. Each record contains five paired original/candidate timings in microseconds, execution order, speedup and three-step correctness diagnostics. The matrix has 224 points and the separate 27B TP2 supplement has 12. Output/state equality is tested on the original and candidate; the dense PyTorch recurrence is a separate BF16-input check.
