#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
RESULTS_DIR="${MAT_IPPO_RESULTS_DIR:-$SCRIPT_DIR/results}"
PACKAGE3_DIR="${MAT_STABILITY_RESULTS_DIR:-$SCRIPT_DIR/../MAT_Stability/results}"
N_RUNS=""
LOOK_FOR_IMPORTS=""
PROTOCOL="all"
MAX_WORKERS=20
PREVIEW=false
OVERWRITE=false

usage() {
    cat <<'EOF'
Usage: Revision/MAT_IPPO_Comparison/launch_tmux.sh [options]

Required:
  --n-runs N                     Number of seed pairs in this batch.
  --look-for-imports true|false  Import Package-3 MAT pairs or generate new pairs.

Options:
  --protocol all|fixed|varying   Default: all.
  --max-workers N                Concurrent persistent tmux slots; default: 20.
  --results-dir PATH             Override the Package-4 result directory.
  --package3-dir PATH            Override the Package-3 result directory.
  --preview                      Print the plan and worker commands; start nothing.
  --overwrite                    Re-run matching jobs and validations.
  --help                         Show this message.

Hyphen and underscore spellings (--n-runs/--n_runs) are both accepted.
Set JULIA_BIN if Julia is not available as `julia` on PATH.
EOF
}

while (($#)); do
    case "$1" in
        --n-runs|--n_runs) N_RUNS="$2"; shift 2 ;;
        --look-for-imports|--look_for_imports) LOOK_FOR_IMPORTS="$2"; shift 2 ;;
        --protocol) PROTOCOL="$2"; shift 2 ;;
        --max-workers|--max_workers) MAX_WORKERS="$2"; shift 2 ;;
        --results-dir|--results_dir) RESULTS_DIR="$2"; shift 2 ;;
        --package3-dir|--package3_dir) PACKAGE3_DIR="$2"; shift 2 ;;
        --preview) PREVIEW=true; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$N_RUNS" ]] || { echo "--n-runs is required." >&2; exit 2; }
[[ -n "$LOOK_FOR_IMPORTS" ]] || { echo "--look-for-imports is required." >&2; exit 2; }
[[ "$N_RUNS" =~ ^[0-9]+$ ]] || { echo "--n-runs must be non-negative." >&2; exit 2; }
[[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "--max-workers must be positive." >&2; exit 2; }
[[ "$LOOK_FOR_IMPORTS" =~ ^(true|false)$ ]] || {
    echo "--look-for-imports must be true or false." >&2; exit 2;
}
[[ "$PROTOCOL" =~ ^(all|fixed|varying)$ ]] || {
    echo "--protocol must be all, fixed, or varying." >&2; exit 2;
}
command -v "$JULIA_BIN" >/dev/null || { echo "Julia executable not found: $JULIA_BIN" >&2; exit 1; }

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
    --n-runs "$N_RUNS"
    --look-for-imports "$LOOK_FOR_IMPORTS"
    --protocol "$PROTOCOL"
    --results-dir "$RESULTS_DIR"
    --package3-dir "$PACKAGE3_DIR"
    --jobs-file "$MANIFEST"
)
[[ "$PREVIEW" == true ]] && PREPARE+=(--preview)
[[ "$OVERWRITE" == true ]] && PREPARE+=(--overwrite)
"${PREPARE[@]}"

mapfile -t JOBS < <(tail -n +2 "$MANIFEST")
if ((${#JOBS[@]} == 0)); then
    echo "No missing jobs for the selected seed pairs."
    exit 0
fi

worker_command() {
    local row="$1" task run_id protocol algorithm result_path command
    IFS=$'\t' read -r task run_id protocol algorithm result_path <<<"$row"
    printf -v command '%q ' \
        "$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT" \
        "$SCRIPT_DIR/run_worker.jl" --task "$task" --run-id "$run_id" \
        --protocol "$protocol" --algorithm "$algorithm" --results-dir "$RESULTS_DIR"
    [[ "$OVERWRITE" == true ]] && command+="--overwrite "
    printf '%s' "$command"
}

session_name() {
    local row="$1" task run_id protocol algorithm result_path
    IFS=$'\t' read -r task run_id protocol algorithm result_path <<<"$row"
    printf 'p4_%s_%s_%s_%s' "$algorithm" "$protocol" "$run_id" "$task"
}

if [[ "$PREVIEW" == true ]]; then
    echo
    echo "Preview only; no plan was persisted and no tmux session was started."
    for row in "${JOBS[@]}"; do
        printf '%s: ' "$(session_name "$row")"
        worker_command "$row"
        echo
    done
    exit 0
fi

command -v tmux >/dev/null || { echo "tmux is required to launch workers." >&2; exit 1; }
WORKER_COUNT=${#JOBS[@]}
((WORKER_COUNT > MAX_WORKERS)) && WORKER_COUNT=$MAX_WORKERS
FIRST_SESSION=""
STARTED_COUNT=0

for ((slot=0; slot<WORKER_COUNT; slot++)); do
    SLOT_SCRIPT="$LAUNCH_DIR/slot_$((slot + 1)).sh"
    session="$(session_name "${JOBS[slot]}")"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -uo pipefail'
        echo 'slot_failed=0'
        for ((index=slot; index<${#JOBS[@]}; index+=WORKER_COUNT)); do
            row="${JOBS[index]}"
            IFS=$'\t' read -r task run_id protocol algorithm result_path <<<"$row"
            job_session="$(session_name "$row")"
            LOG_PATH="$LAUNCH_DIR/${run_id}_${protocol}_${algorithm}_${task}.log"
            command="$(worker_command "$row")"
            printf 'current_session=$(tmux display-message -p -t "$TMUX_PANE" "#S")\n'
            printf 'if [[ "$current_session" != %q ]]; then tmux rename-session -t "$current_session" %q; fi\n' \
                "$job_session" "$job_session"
            printf 'echo %q\n' "Starting $task $run_id/$protocol/$algorithm"
            printf '%s 2>&1 | tee %q || slot_failed=1\n' "$command" "$LOG_PATH"
        done
        echo 'echo "Worker queue finished with status $slot_failed."'
        echo 'exit "$slot_failed"'
    } > "$SLOT_SCRIPT"
    chmod u+x "$SLOT_SCRIPT"
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Skipping active tmux session $session"
        continue
    fi
    tmux new-session -d -s "$session" "bash '$SLOT_SCRIPT'"
    [[ -z "$FIRST_SESSION" ]] && FIRST_SESSION="$session"
    STARTED_COUNT=$((STARTED_COUNT + 1))
    echo "Started $session"
done

echo
echo "Started ${STARTED_COUNT} persistent tmux slots for ${#JOBS[@]} jobs."
echo "They survive SSH disconnects and close automatically after their queues finish."
[[ -n "$FIRST_SESSION" ]] && echo "Attach with: tmux attach -t $FIRST_SESSION"
echo "Logs and manifest: $LAUNCH_DIR"
