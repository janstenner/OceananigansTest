#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
RESULTS_DIR="${MAT_STABILITY_RESULTS_DIR:-$SCRIPT_DIR/results}"
PROTOCOL="all"
PREVIEW=false
WORKER_DRY_RUN=false
OVERWRITE=false

usage() {
    cat <<'EOF'
Usage: Revision/MAT_Stability/launch_tmux.sh [options]

Creates one independent tmux worker per protocol, replicate, and MAT
configuration. A complete launch starts 15 Fixed-IC and 15 Varying-IC workers.

Options:
  --protocol all|fixed|varying  Default: all.
  --results-dir PATH            Override the MAT Stability result directory.
  --preview                     Print the frozen plan and commands; start nothing.
  --dry-run-workers             Start zero-episode verification workers.
  --overwrite                   Re-run already complete matching results.
  --help                        Show this message.

Set JULIA_BIN if Julia is not available as `julia` on PATH.
EOF
}

while (($#)); do
    case "$1" in
        --protocol)
            (($# >= 2)) || { echo "Missing value after --protocol." >&2; exit 2; }
            PROTOCOL="$2"
            shift 2
            ;;
        --results-dir|--results_dir)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            RESULTS_DIR="$2"
            shift 2
            ;;
        --preview) PREVIEW=true; shift ;;
        --dry-run-workers|--dry_run_workers) WORKER_DRY_RUN=true; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$PROTOCOL" =~ ^(all|fixed|varying)$ ]] || {
    echo "--protocol must be all, fixed, or varying." >&2
    exit 2
}
command -v "$JULIA_BIN" >/dev/null || {
    echo "Julia executable not found: $JULIA_BIN" >&2
    exit 1
}

if [[ "$PREVIEW" == true ]]; then
    MANIFEST="$(mktemp)"
    trap 'rm -f "$MANIFEST"' EXIT
else
    LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    LAUNCH_DIR="$RESULTS_DIR/launches/$LAUNCH_ID"
    mkdir -p "$LAUNCH_DIR"
    MANIFEST="$LAUNCH_DIR/jobs.tsv"
fi

PREPARE=(
    "$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT"
    "$SCRIPT_DIR/prepare_runs.jl"
    --protocol "$PROTOCOL"
    --results-dir "$RESULTS_DIR"
    --jobs-file "$MANIFEST"
)
[[ "$WORKER_DRY_RUN" == true ]] && PREPARE+=(--dry-run)
[[ "$PREVIEW" == true ]] && PREPARE+=(--preview)
[[ "$OVERWRITE" == true ]] && PREPARE+=(--overwrite)
"${PREPARE[@]}"

mapfile -t JOBS < <(tail -n +2 "$MANIFEST")
if ((${#JOBS[@]} == 0)); then
    echo "No pending MAT Stability jobs."
    exit 0
fi

worker_command() {
    local row="$1" protocol replicate config run_seed ic_seed episode_target result_path command
    IFS=$'\t' read -r protocol replicate config run_seed ic_seed episode_target result_path <<<"$row"
    local command_parts=(
        "$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT"
        "$SCRIPT_DIR/run_worker.jl"
        --protocol "$protocol"
        --replicate "$replicate"
        --config "$config"
        --results-dir "$RESULTS_DIR"
    )
    [[ "$WORKER_DRY_RUN" == true ]] && command_parts+=(--dry-run)
    [[ "$OVERWRITE" == true ]] && command_parts+=(--overwrite)
    printf -v command '%q ' "${command_parts[@]}"
    printf '%s' "$command"
}

session_name() {
    local row="$1" protocol replicate config run_seed ic_seed episode_target result_path
    IFS=$'\t' read -r protocol replicate config run_seed ic_seed episode_target result_path <<<"$row"
    printf -v replicate '%02d' "$replicate"
    if [[ "$WORKER_DRY_RUN" == true ]]; then
        printf 'mat_dry_%s_%s_%s' "$protocol" "$replicate" "$config"
    else
        printf 'mat_%s_%s_%s' "$protocol" "$replicate" "$config"
    fi
}

if [[ "$PREVIEW" == true ]]; then
    echo
    echo "Preview only; no run plan was persisted and no tmux session was started."
    for row in "${JOBS[@]}"; do
        printf '%s: ' "$(session_name "$row")"
        worker_command "$row"
        echo
    done
    exit 0
fi

command -v tmux >/dev/null || {
    echo "tmux is required to launch workers." >&2
    exit 1
}

FIRST_SESSION=""
STARTED_COUNT=0
for row in "${JOBS[@]}"; do
    IFS=$'\t' read -r protocol replicate config run_seed ic_seed episode_target result_path <<<"$row"
    session="$(session_name "$row")"
    logfile="$LAUNCH_DIR/${protocol}_$(printf '%02d' "$replicate")_${config}.log"
    command="$(worker_command "$row")"
    printf -v quoted_logfile '%q' "$logfile"
    shell_command="set -o pipefail; ${command}2>&1 | tee -a ${quoted_logfile}"
    printf -v quoted_shell_command '%q' "$shell_command"

    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Skipping active tmux session $session"
        continue
    fi
    tmux new-session -d -s "$session" "bash -lc $quoted_shell_command"
    [[ -z "$FIRST_SESSION" ]] && FIRST_SESSION="$session"
    STARTED_COUNT=$((STARTED_COUNT + 1))
    echo "Started $session; log: $logfile"
done

echo
echo "Started $STARTED_COUNT independent tmux workers for ${#JOBS[@]} pending jobs."
echo "They survive SSH disconnects and close when their single job finishes."
[[ -n "$FIRST_SESSION" ]] && echo "Attach with: tmux attach -t $FIRST_SESSION"
echo "Logs and worker manifest: $LAUNCH_DIR"
