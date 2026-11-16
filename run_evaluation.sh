#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
RECOGNITION="robust"
SCOPE="core"
MODEL=""

usage() {
  cat <<'EOF'
Usage: ./run_evaluation.sh [options]

Options:
  --recognition MODE   standard, robust, or extended (default: robust)
  --scope SCOPE        core, aggregate, paper, entity, ablation, or all
                       core creates one evaluation version with all paper tables (default)
                       paper exports standalone Tables 3 and 4 only
                       entity runs cached Wikidata-alias normalization separately
                       all adds entity normalization and every saved ablation model
  --model MODEL        Run ablation only for this saved model directory
  --python PATH        Python executable (default: .venv/bin/python)
  -h, --help           Show this help

Examples:
  ./run_evaluation.sh
  ./run_evaluation.sh --scope paper --recognition standard
  ./run_evaluation.sh --scope entity --recognition robust
  ./run_evaluation.sh --scope ablation --model gpt-5

Each command creates a UTC-timestamped directory under output/evaluation_runs/
and writes a manifest.json containing the configuration and input/output hashes.
Complete evaluation versions use entity_normalization=none. The entity scope is
the separate wikidata_alias_cached sensitivity analysis; it does not silently
change the main paper tables.
EOF
}

while (($#)); do
  case "$1" in
    --recognition)
      RECOGNITION="${2:?--recognition requires a value}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:?--scope requires a value}"
      shift 2
      ;;
    --model)
      MODEL="${2:?--model requires a value}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:?--python requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$RECOGNITION" in
  standard|robust|extended) ;;
  *)
    echo "Invalid recognition mode: $RECOGNITION" >&2
    exit 2
    ;;
esac

case "$SCOPE" in
  core|aggregate|paper|entity|ablation|all) ;;
  *)
    echo "Invalid scope: $SCOPE" >&2
    exit 2
    ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Activate the environment or pass --python PATH." >&2
  exit 2
fi

run_aggregate() {
  "$PYTHON_BIN" -m src.evaluation.run \
    --recognition "$RECOGNITION"
}

run_paper() {
  "$PYTHON_BIN" -m src.evaluation.export_paper_results \
    --recognition "$RECOGNITION"
}

run_entity() {
  "$PYTHON_BIN" -m src.evaluation.analyses.entity_normalization_sensitivity \
    --recognition "$RECOGNITION"
}

run_ablation_dir() {
  local input_dir="$1"
  echo "Running relation-aware ablation: $(basename "$input_dir")"
  "$PYTHON_BIN" -m src.evaluation.analyses.eval_relation_aware_ablation \
    --input-dir "$input_dir" \
    --recognition "$RECOGNITION"
}

run_ablation() {
  local base="$REPO_ROOT/output/relation_aware_ablation"
  if [[ -n "$MODEL" ]]; then
    local selected="$base/$MODEL"
    if [[ ! -d "$selected" ]]; then
      echo "Saved ablation directory not found: $selected" >&2
      exit 2
    fi
    run_ablation_dir "$selected"
    return
  fi

  local found=0
  while IFS= read -r input_dir; do
    found=1
    run_ablation_dir "$input_dir"
  done < <(find "$base" -mindepth 1 -maxdepth 1 -type d -print | LC_ALL=C sort)
  if [[ "$found" -eq 0 ]]; then
    echo "No saved ablation model directories found under $base" >&2
    exit 2
  fi
}

echo "Evaluation scope: $SCOPE"
echo "IDK recognition: $RECOGNITION"
echo "Python: $PYTHON_BIN"

case "$SCOPE" in
  aggregate)
    run_aggregate
    ;;
  paper)
    run_paper
    ;;
  entity)
    run_entity
    ;;
  ablation)
    run_ablation
    ;;
  core)
    run_aggregate
    ;;
  all)
    run_aggregate
    run_entity
    run_ablation
    ;;
esac

echo "Evaluation completed. See: output/evaluation_runs/"
