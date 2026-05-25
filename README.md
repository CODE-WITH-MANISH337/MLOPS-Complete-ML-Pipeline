# MLOPS Complete ML Pipeline (DVC + dvclive + AWS S3)

A reference MLOPS repository that builds an end-to-end **ML pipeline** using **DVC** for data/metric versioning and **dvclive** for experiment tracking. It also includes guidance to store DVC artifacts in **AWS S3**.

---

## What’s included

- **Modular pipeline steps** (run each component individually and/or via DVC)
- **DVC pipeline** (`dvc.yaml`) with support for:
  - running without params
  - running with params via `params.yaml`
- **Experiment tracking** with **dvclive**
- **AWS S3 remote** setup using `dvc[s3]`

---

## Repository workflow (baseline)

1. Create a GitHub repo and clone it locally (add experiments).
2. Add the `src/` folder along with all components.
3. Add `data/`, `models/`, `reports/` to `.gitignore`.
4. Commit and push your initial structure.

---

## DVC pipeline setup (without params)

1. Create `dvc.yaml` and add pipeline stages.
2. Run:
   - `dvc init`
   - `dvc repro` (verify the pipeline automation; check `dvc dag`)
3. Commit and push.

---

## DVC pipeline setup (with params)

1. Add `params.yaml`.
2. In each stage entry point, load parameters from `params.yaml`.
3. Run:
   - `dvc repro` (verifies pipeline + params)
4. Commit and push.

### `params.yaml` (current keys)

```yaml
data_ingestion:
  test_size: 0.2

feature_engineering:
  max_features: 50

model_building:
  n_estimators: 50
  random_state: 2
```

---

## Experiment tracking with DVC + dvclive

1. Install dvclive:

```bash
pip install dvclive
```

2. Add the dvclive block to your entry script (example pattern):

```python
from dvclive import Live
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ... after you compute y_test and predictions

with Live(save_dvc_exp=True) as live:
    live.log_metric('accuracy', accuracy_score(y_test, y_pred))
    live.log_metric('precision', precision_score(y_test, y_pred))
    live.log_metric('recall', recall_score(y_test, y_pred))

    live.log_params(params)  # where params comes from params.yaml
```

3. Run an experiment:

- `dvc exp run`

This creates dvclive experiment outputs (and may generate/refresh `dvc.yaml` artifacts depending on your setup).

4. View experiments:
- `dvc exp show`

5. Manage experiments:
- `dvc exp remove {exp-name}`
- `dvc exp apply {exp-name}`

6. Change `params.yaml`, rerun, and you’ll get new experiments.

---

## AWS S3 remote (dvc store)

### Prerequisites
- AWS account + S3 bucket
- AWS credentials configured locally

### Steps

1. Create an IAM user (with access to S3).
2. Create an S3 bucket.
3. Install:
   - `awscli`
   - `dvc[s3]`

4. Configure AWS credentials:
- `aws configure`

5. Add S3 remote for DVC:
- `dvc remote add -d dvcstore s3://bucketname`

6. When you run experiments, commit/push the experiment outcomes you want to keep:
- `git commit -m "..." && git push`

> Ensure your S3 bucket policy/permissions allow the operations DVC performs (read/write).

---

## How to use this repo

Typical flow:

1. Run components individually (validate each step works end-to-end).
2. Create/maintain `dvc.yaml` stages so `dvc repro` can reproduce results.
3. Use `params.yaml` to control experiments.
4. Use `dvclive` + `dvc exp` to track metrics and compare runs.
5. Use the S3 remote to store DVC outputs in the cloud.

---

## Notes

- Keep generated artifacts (like `data/`, `models/`, `reports/`) out of Git and let DVC manage them.
- Use `dvc dag` to inspect your stage dependency graph.
- If experiments fail, check:
  - DVC remote authentication (S3)
  - parameter loading paths and YAML keys
  - that metrics logging runs only after you compute predictions

---

## License

MIT (see `LICENSE`)
