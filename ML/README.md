# gazelock_ml — Training & Export Pipeline

Internal Python package that produces the refiner `.mlpackage` consumed
by the Swift app.

## One-shot setup

```bash
make ml-setup
```

That runs `uv sync --extra dev` + `uv pip install -e .`. After it, the
CLIs (`gazelock-train`, `gazelock-eval`, `gazelock-export`) are on PATH.

## Smoke run (end-to-end on procedural fixtures)

```bash
make ml-train   # 10 steps; writes weights/debug/final.pt
make ml-eval    # reads that checkpoint, prints PSNR/SSIM/identity/flicker
make ml-export  # writes weights/refiner.mlpackage with L∞ gate
```

No datasets required — everything runs on synthetic fixtures. The
resulting weights are garbage (the model only saw 10 gradient steps on
trivial patterns) but the pipeline is validated end-to-end.

## Real training

1. Download UnityEyes renders. Organise as:
   ```
   <unityeyes_root>/
       identity_0001/0.jpg 0.json 1.jpg 1.json ...
       identity_0002/...
   ```
2. Download FFHQ images (any resolution; 512×512 or 1024×1024 recommended).
3. *(Plumbing for real-data paths is stubbed in `gazelock-train`;
   Phase 2c adds the `--unityeyes-root` + `--ffhq-root` wiring. For
   now train on fixtures; when you're ready, wire up the loaders by
   replacing the `_fixture_source` lambda in `ML/gazelock_ml/cli/train.py`
   with `UnityEyesDataset(...).iter_identity(...)` compositions.)*

## Tests

```bash
make ml-test       # fast tests only
make ml-test-slow  # includes Core ML export test (~30–60 s)
```

## License

MIT (see repository `LICENSE`). Note the UnityEyes-data licensing
constraint documented in the top-level README; it applies when
shipping trained weights.
