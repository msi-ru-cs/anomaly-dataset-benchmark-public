# src/main.py
import os
import sys
import argparse
import importlib
from pathlib import Path
import yaml

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _deep_update(a: dict, b: dict):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a

def get_cfg(cfg: dict, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def set_cfg(cfg: dict, path: str, value):
    cur = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def has_prepared_csvs(prepared_dir: str) -> bool:
    p = Path(prepared_dir) if prepared_dir else None
    return bool(p and p.exists() and any(p.rglob("*.csv")))

def merge_sidecar_cfgs(cfg_path: str) -> dict:
    """Load config.yaml and merge nab.yaml / ms.yaml / exa.yaml / prep.yaml if present (non-Hydra composition)."""
    cfg = load_cfg(cfg_path)
    cfg_dir = Path(cfg_path).parent
    for name in ("nab.yaml", "ms.yaml", "exa.yaml", "prep.yaml"):
        p = cfg_dir / name
        if p.exists():
            side = load_cfg(str(p))
            _deep_update(cfg, side)
            print(f"[CFG] merged {name}")

    # Provide a default prepared_dir if missing
    if not get_cfg(cfg, "dataset.prepared_dir"):
        ds = str(get_cfg(cfg, "dataset.name", "nab")).lower()
        set_cfg(cfg, "dataset.prepared_dir", f"output/prepared/{ds}")
        print(f"[CFG] set dataset.prepared_dir=output/prepared/{ds}")

    return cfg

def _resolve_and_write_cfg(cfg: dict, orig_path: str) -> str:
    """
    Resolve simple ${dataset.name} placeholders that our trainer won't expand,
    then write a sibling _resolved.yaml for the trainer to consume.
    """
    ds = str(get_cfg(cfg, "dataset.name", "nab")).lower()

    # Resolve dataset.prepared_dir
    pd = get_cfg(cfg, "dataset.prepared_dir", "")
    if isinstance(pd, str) and "${dataset.name}" in pd:
        pd_resolved = pd.replace("${dataset.name}", ds)
        set_cfg(cfg, "dataset.prepared_dir", pd_resolved)
        print(f"[CFG] resolved dataset.prepared_dir={pd_resolved}")

    # Resolve output.tag (nice to have)
    tag = get_cfg(cfg, "output.tag", "")
    if isinstance(tag, str) and "${dataset.name}" in tag:
        tag_resolved = tag.replace("${dataset.name}", ds)
        set_cfg(cfg, "output.tag", tag_resolved)
        print(f"[CFG] resolved output.tag={tag_resolved}")

    # Optionally resolve any nested strings that include ${dataset.name}
    def _resolve_inplace(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = _resolve_inplace(v)
        elif isinstance(obj, list):
            return [_resolve_inplace(v) for v in obj]
        elif isinstance(obj, str) and "${dataset.name}" in obj:
            return obj.replace("${dataset.name}", ds)
        return obj
    _resolve_inplace(cfg)

    resolved_path = str(Path(orig_path).with_name("_resolved.yaml"))
    with open(resolved_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return resolved_path

def run_prep_if_needed(cfg: dict, cfg_path: str):
    """
    Optional dataset prep step.

    Looks for module: src.prep.dataprep_<dataset> (e.g., dataprep_nab),
    else falls back to src.prep.dataprep.

    Accepted entrypoints (in order):
      - run_preprocess(cfg)
      - run_dataprep(cfg)
      - prepare_dataset(cfg)
      - run(cfg)
      - main(cfg) or main(cfg_path)
    """
    enabled = bool(get_cfg(cfg, "steps.prep.enabled", False))
    overwrite = bool(get_cfg(cfg, "steps.prep.overwrite", False))
    prepared_dir = get_cfg(cfg, "dataset.prepared_dir")

    if not enabled:
        print("[PREP] disabled; skipping.")
        return

    if prepared_dir and (not overwrite) and has_prepared_csvs(prepared_dir):
        print(f"[PREP] found CSVs in {prepared_dir}; skipping (set steps.prep.overwrite=true to force).")
        return

    dataset = str(get_cfg(cfg, "dataset.name", "nab")).lower()
    module_name = f"src.prep.dataprep_{dataset}"
    try:
        mod = importlib.import_module(module_name)
        print(f"[PREP] using module: {module_name}")
    except ImportError:
        mod = importlib.import_module("src.prep.dataprep")
        print("[PREP] dataset-specific prep not found; using src.prep.dataprep")

    candidate_names = ["run_preprocess", "run_dataprep", "prepare_dataset", "run", "main"]

    cfg_args = []
    try:
        from omegaconf import OmegaConf  # available via hydra-core
        cfg_args.append(OmegaConf.create(cfg))
    except Exception:
        pass
    cfg_args.append(cfg)       # raw dict
    cfg_args.append(cfg_path)  # some modules accept a path

    for name in candidate_names:
        if hasattr(mod, name):
            fn = getattr(mod, name)
            for arg in cfg_args:
                try:
                    fn(arg)
                    return
                except TypeError:
                    continue

    raise RuntimeError(
        f"[PREP] No usable entrypoint found in {mod.__name__} "
        f"(tried {', '.join(candidate_names)})."
    )

def run_trainer(cfg_path: str, tag: str = None):
    from src.pipelines.train_tf import main as tf_main
    # mimic trainer CLI
    sys.argv = ["train_tf.py", "--config", cfg_path] + (["--tag", tag] if tag else [])
    tf_main()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"[FATAL] config not found: {args.config}")
        sys.exit(2)

    cfg = merge_sidecar_cfgs(args.config)

    print(f"[CWD] {Path.cwd()}")
    print(f"[CFG] dataset={get_cfg(cfg, 'dataset.name', 'unknown')}")

    # 1) Optional prep
    run_prep_if_needed(cfg, args.config)

    # 2) Resolve ${dataset.name} and write a trainer-friendly config
    resolved_cfg_path = _resolve_and_write_cfg(cfg, args.config)

    # 3) Train (TF)
    run_trainer(resolved_cfg_path, args.tag)

if __name__ == "__main__":
    main()
