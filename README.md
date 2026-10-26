# Filter runs and feature tracks

This branch holds data, not code. It has the filter runs behind every table
in the README on `main` (`runs/`, one `.npz` per variant and flight) and the
cached KLT feature tracks, including the tracks with the camera switched off
(`cache/tracks/`). They come from the run of 2026-10-18 at commit `f94bb2d`.

To use them, copy both folders into a checkout of `main`:

```bash
git fetch origin runs
git archive origin/runs runs cache | tar -x
```

`scripts/reproduce.sh` then skips every filter run and goes straight to
scoring.
