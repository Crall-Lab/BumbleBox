BumbleBox Local Tracking Index
==============================

This folder is the default lightweight local index for BumbleBox run summaries,
tracking CSVs, and tracking-optimization artifacts.

It is intentionally repo-local so each checkout/user can keep their own index in
a predictable place. Generated contents in this folder are ignored by Git; this
README is kept so the folder exists and the convention is visible.

Expected generated contents:
- runs/YYYY-MM-DD/<session_name>/
- tracking/YYYY-MM-DD/<session_name>/
- optimization/YYYY-MM-DD/<optimization_run>/
- posthoc/YYYY-MM-DD/<posthoc_run>/

Large raw videos are intentionally not copied here.
