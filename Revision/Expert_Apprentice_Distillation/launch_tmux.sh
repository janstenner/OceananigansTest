#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JULIA_BIN="${JULIA_BIN:-julia}"
SYSTEMD_INHIBIT_BIN="${SYSTEMD_INHIBIT_BIN:-systemd-inhibit}"
SYSTEMD_INHIBIT_WHAT="${SYSTEMD_INHIBIT_WHAT:-sleep:idle:shutdown}"
WORKER_DIR="${DISTILLATION_WORKER_DIRECTORY:-$SCRIPT_DIR/worker_results}"
PROTOCOL="varying"
SPLIT="train"
MAX_WORKERS=40
FIXED_EXPERT_PATH=""
VARYING_EXPERT_PATH=""
RUN_SEED=600600
PREVIEW=false
OVERWRITE=false
ALLOW_FRESH_EXPERT=false
USE_SYSTEMD_INHIBIT=true

usage() {
    cat <<'EOF'
Usage: Revision/Expert_Apprentice_Distillation/launch_tmux.sh [options]

Defaults to the 40 Varying-IC training workers: 20 basis ICs x 2 mirrors.
Each Varying-IC worker evaluates all 96 horizontal offsets. Fixed IC creates
one worker with one episode.

Options:
  --protocol all|fixed|varying       Default: varying.
  --split all|train|validation|test  Default: train; ignored by Fixed IC.
  --max-workers N                    Persistent tmux slots; default: 40.
  --worker-dir PATH                  Worker JLD2/result root.
  --fixed-expert-path PATH           Fixed-IC expert checkpoint.
  --varying-expert-path PATH         Varying-IC expert checkpoint.
  --run-seed N                       Initialization seed; default: 600600.
  --preview                          Print jobs and commands; start nothing.
  --overwrite                        Regenerate matching worker files.
  --allow-fresh-expert               Smoke tests only; no production expert.
  --no-systemd-inhibit               Local/debug use only.
  --help                             Show this message.
EOF
}

while (($#)); do
    case "$1" in
        --protocol) PROTOCOL="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --max-workers|--max_workers) MAX_WORKERS="$2"; shift 2 ;;
        --worker-dir|--worker_dir) WORKER_DIR="$2"; shift 2 ;;
        --fixed-expert-path|--fixed_expert_path) FIXED_EXPERT_PATH="$2"; shift 2 ;;
        --varying-expert-path|--varying_expert_path) VARYING_EXPERT_PATH="$2"; shift 2 ;;
        --run-seed|--run_seed) RUN_SEED="$2"; shift 2 ;;
        --preview) PREVIEW=true; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        --allow-fresh-expert) ALLOW_FRESH_EXPERT=true; shift ;;
        --no-systemd-inhibit|--no-inhibit) USE_SYSTEMD_INHIBIT=false; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$PROTOCOL" =~ ^(all|fixed|varying)$ ]] || {
    echo "--protocol must be all, fixed, or varying." >&2; exit 2;
}
[[ "$SPLIT" =~ ^(all|train|validation|test)$ ]] || {
    echo "--split must be all, train, validation, or test." >&2; exit 2;
}
[[ "$MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--max-workers must be positive." >&2; exit 2;
}
[[ "$RUN_SEED" =~ ^[0-9]+$ ]] || {
    echo "--run-seed must be non-negative." >&2; exit 2;
}
command -v "$JULIA_BIN" >/dev/null || {
    echo "Julia executable not found: $JULIA_BIN" >&2; exit 1;
}
if [[ "$USE_SYSTEMD_INHIBIT" == true && "$PREVIEW" == false ]]; then
    command -v "$SYSTEMD_INHIBIT_BIN" >/dev/null || {
        echo "systemd-inhibit executable not found: $SYSTEMD_INHIBIT_BIN" >&2
        echo "Use --no-systemd-inhibit only when inhibition is intentionally unnecessary." >&2
        exit 1
    }
fi

if [[ "$PREVIEW" == true ]]; then
    MANIFEST="$(mktemp)"
    trap 'rm -f "$MANIFEST"' EXIT
    LAUNCH_DIR=""
else
    LAUNCH_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    LAUNCH_DIR="$WORKER_DIR/launches/$LAUNCH_ID"
    mkdir -p "$LAUNCH_DIR"
    MANIFEST="$LAUNCH_DIR/jobs.tsv"
fi

PREPARE=(
    "$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT"
    "$SCRIPT_DIR/prepare_corpus_workers.jl"
    --protocol "$PROTOCOL" --split "$SPLIT"
    --worker-dir "$WORKER_DIR" --jobs-file "$MANIFEST" --run-seed "$RUN_SEED"
)
[[ -n "$FIXED_EXPERT_PATH" ]] && PREPARE+=(--fixed-expert-path "$FIXED_EXPERT_PATH")
[[ -n "$VARYING_EXPERT_PATH" ]] && PREPARE+=(--varying-expert-path "$VARYING_EXPERT_PATH")
[[ "$ALLOW_FRESH_EXPERT" == true ]] && PREPARE+=(--allow-fresh-expert)
[[ "$OVERWRITE" == true ]] && PREPARE+=(--overwrite)
"${PREPARE[@]}"

mapfile -t JOBS < <(tail -n +2 "$MANIFEST")
if ((${#JOBS[@]} == 0)); then
    echo "No missing distillation corpus workers."
    exit 0
fi

worker_command() {
    local row="$1" protocol split base_seed mirror output_path expert_path command
    IFS=$'\t' read -r protocol split base_seed mirror output_path <<<"$row"
    local command_parts=(
        "$JULIA_BIN" --startup-file=no "--project=$PROJECT_ROOT"
        "$SCRIPT_DIR/run_corpus_worker.jl"
        --protocol "$protocol" --worker-dir "$WORKER_DIR" --run-seed "$RUN_SEED"
    )
    if [[ "$protocol" == varying ]]; then
        command_parts+=(--split "$split" --base-seed "$base_seed" --mirror "$mirror")
        expert_path="$VARYING_EXPERT_PATH"
    else
        expert_path="$FIXED_EXPERT_PATH"
    fi
    [[ -n "$expert_path" ]] && command_parts+=(--expert-path "$expert_path")
    [[ "$OVERWRITE" == true ]] && command_parts+=(--overwrite)
    [[ "$ALLOW_FRESH_EXPERT" == true ]] && command_parts+=(--allow-fresh-expert)
    if [[ "$USE_SYSTEMD_INHIBIT" == true ]]; then
        command_parts=(
            "$SYSTEMD_INHIBIT_BIN"
            "--what=$SYSTEMD_INHIBIT_WHAT"
            "--who=Oceananigans distillation $protocol $split $base_seed $mirror"
            "--why=RBC expert-apprentice distillation corpus worker"
            "--mode=block"
            "${command_parts[@]}"
        )
    fi
    printf -v command '%q ' "${command_parts[@]}"
    printf '%s' "$command"
}

session_name() {
    local row="$1" protocol split base_seed mirror output_path
    IFS=$'\t' read -r protocol split base_seed mirror output_path <<<"$row"
    if [[ "$protocol" == fixed ]]; then
        printf 'distill_fixed'
    else
        printf 'distill_%s_%s_m%s' "$split" "$base_seed" "${mirror:0:1}"
    fi
}

if [[ "$PREVIEW" == true ]]; then
    echo
    echo "Preview only; no tmux session was started."
    for row in "${JOBS[@]}"; do
        printf '%s: ' "$(session_name "$row")"
        worker_command "$row"
        echo
    done
    exit 0
fi

command -v tmux >/dev/null || { echo "tmux is required." >&2; exit 1; }
WORKER_COUNT=${#JOBS[@]}
((WORKER_COUNT > MAX_WORKERS)) && WORKER_COUNT=$MAX_WORKERS
FIRST_SESSION=""
STARTED_COUNT=0

for ((slot=0; slot<WORKER_COUNT; slot++)); do
    SLOT_SCRIPT="$LAUNCH_DIR/slot_$((slot + 1)).sh"
    first_row="${JOBS[slot]}"
    session="$(session_name "$first_row")"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -uo pipefail'
        echo 'slot_failed=0'
        for ((index=slot; index<${#JOBS[@]}; index+=WORKER_COUNT)); do
            row="${JOBS[index]}"
            IFS=$'\t' read -r protocol split base_seed mirror output_path <<<"$row"
            job_name="$(session_name "$row")"
            log_path="$LAUNCH_DIR/${job_name}.log"
            command="$(worker_command "$row")"
            printf 'current_session=$(tmux display-message -p -t "$TMUX_PANE" "#S")\n'
            printf 'if [[ "$current_session" != %q ]]; then tmux rename-session -t "$current_session" %q; fi\n' "$job_name" "$job_name"
            printf 'echo %q\n' "Starting $job_name"
            printf '%s 2>&1 | tee %q || slot_failed=1\n' "$command" "$log_path"
        done
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
echo "Started $STARTED_COUNT persistent tmux slots for ${#JOBS[@]} jobs."
echo "Manifest and logs: $LAUNCH_DIR"
[[ -n "$FIRST_SESSION" ]] && echo "Attach with: tmux attach -t $FIRST_SESSION"
