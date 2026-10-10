# fy

Thin command-line glue: wrappers only, no domain logic. It finds the repo root by walking up to
`CONVENTIONS.md`, so it works from any directory.

```
fy build [options]            build freyja.xml and params/snapshot.* (pre_processor options: --dry-run, --xlsx FILE ...)
fy watch [options]            rebuild whenever the sheet or template changes
fy test [pytest options]      pytest over tests/ and checks/
fy check [--tier gate|advisory]   the model checks on the committed model and snapshot; exit 1 on a gate failure
                              (a full run also writes checks/last_run.json, which run records quote)
fy run <scenario>             run sim/scenarios/<scenario>.yaml; writes sim/runs/<id>/summary.json (+ raw.npz, untracked)
fy view <scenario> [--set k=v]   watch a scenario live in the MuJoCo viewer (no record written)
fy runs list [--scenario S]   the recorded runs
fy runs show <id>             one run in full
fy runs compare <idA> <idB>   the inputs that differ, then the metric deltas
fy runs digest                write sim/runs/DIGEST.md
fy runs envelope              write sim/runs/ENVELOPE.md
```

Run it as `python tools/fy <command>`, or put `tools\fy` on your PATH and use `fy <command>` (Windows,
`fy.cmd` picks the repo `.venv` when it exists).
