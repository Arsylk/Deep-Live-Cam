# Pipeline baselines

The active webcam/INSWapper-128 baseline is registered in
`webcam-inswapper128.json`. Private video remains local under the ignored
`arch-linux/runtime/.../benchmarks` directory; the registry keeps the hashes,
configuration identity, and summary metrics needed to detect substitution.

Record a future Arch candidate without opening another camera or stream:

```bash
.venv/bin/python tools/pipeline_baseline.py record \
  --id candidate-name --role candidate --frames 50 --sample-fps 5
```

Compare an analyzed candidate with the active baseline:

```bash
.venv/bin/python tools/pipeline_baseline.py compare \
  --baseline active --candidate PATH/TO/CANDIDATE \
  --output PATH/TO/CANDIDATE/vs-active.json
```

The Windows service uses the same passive recorder after the updated
`modules/pipeline_benchmark.py`, `windows/run_network.py`, and activation files
are deployed. Its request file is
`windows/runtime/network-live/benchmarks/request.json`; run the same tool on
Windows with `--state-dir windows/runtime/network-live`.

A separately performed live take is useful for diagnosis but is not a fair
winner/loser test. For a definitive Arch-versus-Windows or before-versus-after
result, replay the exact frozen `reference.mkv` from the active baseline through
both candidates. The comparator checks a container-independent SHA-256 of the
ordered decoded BGR frames, frame count, cadence, resolution, and source-identity
hash and reports `diagnostic-only` when they differ.

The comparison is a metric vector, not one opaque score. Critical face quality
dimensions are non-inferiority gates; speed cannot compensate for worse seam,
flicker, tracking, or face-stability results. Full-reference metrics are labeled
as signal preservation because the intended identity edit itself lowers them.
