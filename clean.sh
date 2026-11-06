#!/usr/bin/env bash
# Remove ignored development artifacts; keep experiment data and environments.
set -euo pipefail

usage() {
    echo "Usage: ./clean.sh [--dry-run] [--logs]"
    echo "  --dry-run  Preview without deleting files."
    echo "  --logs     Also remove ignored *.log files at the repository root."
}

mode=-fdX
logs=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) mode=-ndX ;;
        --logs) logs=true ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
git rev-parse --show-toplevel >/dev/null

# git clean -X only removes ignored files and preserves tracked files.
# Use an explicit allowlist: many ignored directories hold valuable results.
paths=(
    .cache .pytest_cache .mypy_cache .ruff_cache .hypothesis
    __pycache__ build dist htmlcov docs/_build
    .coverage coverage.xml nosetests.xml
)
shopt -s nullglob
paths+=(.coverage.* *.pyc *.pyo)
for source in mineral plotter scripts tests examples; do
    [[ -d "$source" ]] || continue
    while IFS= read -r -d '' artifact; do
        paths+=("$artifact")
    done < <(find "$source" -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name .ipynb_checkpoints \) -prune -print0)
done
if "$logs"; then
    paths+=(*.log)
fi

for artifact in "${paths[@]}"; do
    [[ -e "$artifact" || -L "$artifact" ]] || continue
    git clean "$mode" -- ":(literal)$artifact"
done
