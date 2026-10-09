import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
import yaml

def create_run_dir(root, name, seed):
    label = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
    run = Path(root) / 'results' / f'{label}_{name}_seed{seed}'
    for child in ['config', 'checkpoints', 'raw_metrics', 'figures', 'logs', 'summary']:
        (run / child).mkdir(parents=True, exist_ok=True)
    return run

def save_metadata(run, config, root):
    (run / 'config' / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    def git(*args):
        p = subprocess.run(['git', *args], cwd=root, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else 'unknown'
    metadata = dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain')),
                    created=dt.datetime.now(dt.timezone.utc).isoformat())
    (run / 'config' / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))

def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
