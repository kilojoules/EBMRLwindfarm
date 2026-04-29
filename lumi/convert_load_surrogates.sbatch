#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=conv_surr
#SBATCH --output=logs/conv_surr_%j.out
#SBATCH --error=logs/conv_surr_%j.err

set -uo pipefail
# Note: -e disabled. WindGym audit can fail (missing optional deps); we want
# the conversion to run regardless. Conversion failure is checked explicitly.
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

# 0. Ensure conversion deps in pixi env.
pixi run python -c "import torch" 2>/dev/null || pixi run install-torch-rocm
pixi run pip install --quiet "tensorflow-cpu" "keras" "joblib" "scikit-learn" 2>&1 | tail -3 || \
    pixi run pip install --quiet "tensorflow" "keras" "joblib" "scikit-learn"

# 1. Pull latest WindGym (preserves local speed mods if uncommitted only as
#    diff; refuses if commit conflict — fix manually then rerun).
WG_DIR=/scratch/project_465002609/julian/windgym
echo "=== WindGym audit ==="
git -C "$WG_DIR" fetch origin --quiet
echo "Local HEAD:  $(git -C "$WG_DIR" rev-parse --short HEAD)"
echo "Origin main: $(git -C "$WG_DIR" rev-parse --short origin/main)"
echo "Diff vs origin/main (speed mods etc.):"
git -C "$WG_DIR" log --oneline origin/main..HEAD || true
git -C "$WG_DIR" diff --stat origin/main || true
echo "Pulling..."
git -C "$WG_DIR" pull --rebase --autostash origin main || {
    echo "WARN: pull failed; continuing with current HEAD"; }
git -C "$WG_DIR" log --oneline -5
echo

# 2. Reinstall WindGym + copy mods (matches your other sbatch pattern).
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

# 3. Sanity: WindGym exposes a flow-sampling API? (best-effort, non-fatal)
pixi run python - <<'PY' || echo "WARN: WindGym import failed; skipping audit. See err log."
import inspect, importlib
try:
    import WindGym
except Exception as e:
    print(f"WindGym import error: {e}")
    raise SystemExit(0)
candidates = ("get_flow_at", "flow_field", "sample_flow", "rotor_inflow")
attrs = [a for a in dir(WindGym) if not a.startswith("_")]
print("WindGym top-level:", attrs[:30])
try:
    from WindGym import WindFarmEnv as Env
except Exception as e:
    print(f"WindFarmEnv import error: {e}")
    Env = None
if Env is not None:
    methods = [m for m in dir(Env) if not m.startswith("_")]
    for c in candidates:
        print(f"  {c}: {'YES' if c in methods else 'no'}")
PY

# 4. Convert keras -> torch. Fatal if this fails.
SURR_SRC=/scratch/project_465002609/julian/Teodor_surrogates
OUT=$PROJECT_DIR/checkpoints/teodor_dlc12_torch.pt
mkdir -p "$(dirname "$OUT")"
set -e
pixi run python -u scripts/convert_load_surrogates.py \
    --src "$SURR_SRC" --out "$OUT"

# 5. Smoke test: load with pure torch, predict at yaw=0 vs yaw=20.
pixi run python -u scripts/probe_flap_del.py --bundle "$OUT"
