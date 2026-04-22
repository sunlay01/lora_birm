## Environment Snapshot

This repository includes two dependency files:

- `requirements.txt`: minimal project dependencies
- `requirements.lock.txt`: full package snapshot from the current machine via `python3 -m pip freeze`

Snapshot details:

- Date: 2026-04-22
- Python: 3.10.8
- Source environment: current runtime on the training machine

Recreate approximately with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

If the full lock file is too strict for another machine, start from `requirements.txt` first.
