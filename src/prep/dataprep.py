# src/prep/dataprep.py
"""
Generic dataprep dispatcher used as a fallback.

Note:
- main.py FIRST tries to import src.prep.dataprep_<dataset> (e.g., dataprep_nab, dataprep_ms).
- If that import fails, it falls back to THIS module and calls run_dataprep(cfg).
- We therefore keep a small registry here so both "nab" and "ms" can still run via the fallback.
"""

from typing import Any
try:
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    DictConfig = Any  # type: ignore

# Always include NAB
from .dataprep_nab import run_preprocess as nab_run

# MS is optional (repo may not have it yet); guard the import
try:
    from .dataprep_ms import run_preprocess as ms_run  # type: ignore
except Exception:  # pragma: no cover
    ms_run = None  # type: ignore

# >>> NEW: Exathlon (guarded import)
try:
    from .dataprep_exathlon import run_preprocess as exa_run  # type: ignore
except Exception:  # pragma: no cover
    exa_run = None  # type: ignore
# <<<

REGISTRY = {"nab": nab_run}
if ms_run:
    REGISTRY["ms"] = ms_run

# >>> NEW: register exathlon (and a short alias)
if exa_run:
    REGISTRY["exathlon"] = exa_run
    REGISTRY["exa"] = exa_run
# <<<

def _get_dataset_name(cfg) -> str:
    try:
        return str(cfg.dataset.name).lower()
    except Exception:
        try:
            return str(cfg.get("dataset", {}).get("name", "nab")).lower()
        except Exception:
            return "nab"

def run_dataprep(cfg: DictConfig):
    name = _get_dataset_name(cfg)
    if name not in REGISTRY:
        raise ValueError(f"Unsupported dataset '{name}'. Known: {list(REGISTRY)}")
    return REGISTRY[name](cfg)

prepare_dataset = run_dataprep
run = run_dataprep
main = run_dataprep
