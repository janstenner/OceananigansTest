#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
RESULTS_DIR="${PACKAGE11_RESULTS_DIR:-$SCRIPT_DIR/results}"
COMPARISON_DIR="$SCRIPT_DIR/../MAT_IPPO_Comparison/results"
PACKAGE7_RESULTS="$SCRIPT_DIR/../Package7/results/260830_173924"
PACKAGE8_RESULTS="$SCRIPT_DIR/../Package8/results/260830_231109"
PROTOCOL=all
GROUPING=all
PREVIEW=false
OPENBLAS_THREADS=3
OMP_THREADS=1

usage() {
    cat <<'EOF'
Usage: bash Revision/MaskedTraining/launch_tmux.sh [options]

Starts 40 detached, self-closing training sessions at Ra=1e4:
10 Comparison seed pairs x Fixed/Varying x GC/SC. No dense retraining.
Completed runs and active sessions are skipped; failed runs restart from seed.

  --protocol all|fixed|varying
  --grouping all|gc|sc
  --results-dir PATH
  --comparison-dir PATH
  --package7-results PATH
  --package8-results PATH
  --openblas-threads N          Default: 3.
  --omp-threads N               Default: 1.
  --preview                    Validate inputs and print commands; start nothing.
  --help
EOF
}

while (($#)); do
    case "$1" in
        --preview) PREVIEW=true; shift ;;
        --help) usage; exit 0 ;;
        --protocol|--grouping|--results-dir|--comparison-dir|--package7-results|--package8-results|--openblas-threads|--omp-threads)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            case "$1" in
                --protocol) PROTOCOL="$2" ;;
                --grouping) GROUPING="$2" ;;
                --results-dir) RESULTS_DIR="$2" ;;
                --comparison-dir) COMPARISON_DIR="$2" ;;
                --package7-results) PACKAGE7_RESULTS="$2" ;;
                --package8-results) PACKAGE8_RESULTS="$2" ;;
                --openblas-threads) OPENBLAS_THREADS="$2" ;;
                --omp-threads) OMP_THREADS="$2" ;;
            esac
            shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
[[ "$PROTOCOL" =~ ^(all|fixed|varying)$ ]] || { echo "Invalid protocol." >&2; exit 2; }
[[ "$GROUPING" =~ ^(all|gc|sc)$ ]] || { echo "Invalid grouping." >&2; exit 2; }
[[ "$OPENBLAS_THREADS" =~ ^[1-9][0-9]*$ && "$OMP_THREADS" =~ ^[1-9][0-9]*$ ]] || {
    echo "Thread counts must be positive." >&2; exit 2;
}
export OPENBLAS_NUM_THREADS="$OPENBLAS_THREADS" OMP_NUM_THREADS="$OMP_THREADS"
command -v "$JULIA_BIN" >/dev/null || { echo "Julia not found: $JULIA_BIN" >&2; exit 1; }
if [[ "$PREVIEW" == true ]]; then
    MANIFEST="$(mktemp)"
    trap 'rm -f "$MANIFEST"' EXIT
else
    command -v tmux >/dev/null || { echo "tmux is required." >&2; exit 1; }
    mkdir -p "$RESULTS_DIR"
    RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
    LAUNCH_DIR="$RESULTS_DIR/launches/$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mkdir -p "$LAUNCH_DIR"
    MANIFEST="$LAUNCH_DIR/jobs.tsv"
fi
PREPARE=("$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT" "$SCRIPT_DIR/prepare_runs.jl"
    --results-dir "$RESULTS_DIR" --comparison-dir "$COMPARISON_DIR"
    --package7-results "$PACKAGE7_RESULTS" --package8-results "$PACKAGE8_RESULTS"
    --protocol "$PROTOCOL" --grouping "$GROUPING" --jobs-file "$MANIFEST")
[[ "$PREVIEW" == true ]] && PREPARE+=(--preview)
"${PREPARE[@]}"
mapfile -t JOBS < <(tail -n +2 "$MANIFEST")
started=0
first_session=""
for row in "${JOBS[@]}"; do
    IFS=$'\t' read -r protocol grouping run_id <<<"$row"
    session="p11_${protocol}_${grouping}_${run_id}"
    COMMAND=("$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT" "$SCRIPT_DIR/run_worker.jl"
        --results-dir "$RESULTS_DIR" --protocol "$protocol" --grouping "$grouping" --run-id "$run_id")
    printf -v command '%q ' "${COMMAND[@]}"
    if [[ "$PREVIEW" == true ]]; then
        printf '%s: %s\n' "$session" "$command"
        continue
    fi
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Skipping active session $session"
        continue
    fi
    SLOT_SCRIPT="$LAUNCH_DIR/$session.sh"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'export OPENBLAS_NUM_THREADS=%q OMP_NUM_THREADS=%q\n' "$OPENBLAS_THREADS" "$OMP_THREADS"
        printf 'cd %q\n' "$PROJECT_ROOT"
        printf '%s 2>&1 | tee %q\n' "$command" "$LAUNCH_DIR/$session.log"
    } > "$SLOT_SCRIPT"
    printf -v shell_command 'bash %q' "$SLOT_SCRIPT"
    tmux new-session -d -s "$session" "$shell_command"
    started=$((started + 1))
    [[ -n "$first_session" ]] || first_session="$session"
    echo "Started $session"
done
if [[ "$PREVIEW" == true ]]; then
    echo "Preview: ${#JOBS[@]} training sessions; no frozen manifest or tmux sessions created."
else
    echo "Started $started persistent training workers. Logs: $LAUNCH_DIR"
    [[ -z "$first_session" ]] || echo "Attach: tmux attach -t $first_session"
fi
