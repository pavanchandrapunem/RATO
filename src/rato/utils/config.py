"""Configuration loading with explicit paper-vs-assumption provenance."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]

def load_config(root=ROOT):
    root = Path(root)
    files = dict(paper='paper_parameters.yaml', assumptions='implementation_assumptions.yaml',
                 environment='environment.yaml', training='training.yaml')
    return {k: yaml.safe_load((root / 'configs' / v).read_text()) for k, v in files.items()}

def configured(root=ROOT, **changes):
    cfg = deepcopy(load_config(root))
    for key, val in changes.items():
        cfg['environment' if key in cfg['environment'] else 'training'][key] = val
    return cfg
