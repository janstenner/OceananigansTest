#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
runner="${script_directory}/run_worker.jl"
results_directory="${MAT_STABILITY_RESULTS_DIR:-${script_directory}/results}"
julia_binary="${JULIA_BIN:-julia}"
systemd_inhibit_binary="${SYSTEMD_INHIBIT_BIN:-systemd-inhibit}"
systemd_inhibit_what="${SYSTEMD_INHIBIT_WHAT:-sleep:idle:shutdown}"

preview=false
worker_dry_run=false
overwrite=false
use_systemd_inhibit=true
protocol_selection=all

usage() {
    cat <<'EOF'
Usage: launch_tmux.sh [--protocol all|fixed|varying] [options]

Starts five detached tmux sessions per selected protocol.
The default selection "all" starts:
  mat_fixed_01 ... mat_fixed_05
  mat_varying_01 ... mat_varying_05

Options:
  --protocol VALUE   Select all, fixed, or varying workers (default: all).
  --preview          Print the planned sessions without starting them.
  --dry-run-workers  Start zero-episode verification workers in separate sessions.
  --overwrite        Pass --overwrite to every worker.
  --no-systemd-inhibit
                     Start workers without a systemd inhibitor (local/debug only).
  --help             Show this message.

Environment:
  JULIA_BIN                 Julia executable (default: julia)
  MAT_STABILITY_RESULTS_DIR Result root
  SYSTEMD_INHIBIT_BIN       systemd-inhibit executable (default: systemd-inhibit)
  SYSTEMD_INHIBIT_WHAT      Locks to request (default: sleep:idle:shutdown)
EOF
}

while (($#)); do
    case "$1" in
        --preview)
            preview=true
            ;;
        --dry-run-workers)
            worker_dry_run=true
            ;;
        --overwrite)
            overwrite=true
            ;;
        --no-systemd-inhibit|--no-inhibit)
            use_systemd_inhibit=false
            ;;
        --protocol)
            (($# >= 2)) || {
                echo "Missing value after --protocol." >&2
                exit 2
            }
            protocol_selection="$2"
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

case "${protocol_selection}" in
    all)
        protocols=(fixed varying)
        ;;
    fixed)
        protocols=(fixed)
        ;;
    varying)
        protocols=(varying)
        ;;
    *)
        echo "Invalid protocol '${protocol_selection}'; use all, fixed, or varying." >&2
        exit 2
        ;;
esac

command -v tmux >/dev/null 2>&1 || {
    echo "tmux was not found in PATH." >&2
    exit 1
}
command -v "${julia_binary}" >/dev/null 2>&1 || {
    echo "Julia executable '${julia_binary}' was not found." >&2
    exit 1
}
if [[ "${use_systemd_inhibit}" == true && "${preview}" == false ]]; then
    command -v "${systemd_inhibit_binary}" >/dev/null 2>&1 || {
        echo "systemd-inhibit executable '${systemd_inhibit_binary}' was not found." >&2
        echo "Use --no-systemd-inhibit only if inhibition is intentionally unnecessary." >&2
        exit 1
    }
fi

log_directory="${results_directory}/logs"
mkdir -p "${log_directory}"
first_session=""

for protocol in "${protocols[@]}"; do
    for replicate in 1 2 3 4 5; do
        printf -v replicate_padded "%02d" "${replicate}"
        if [[ "${worker_dry_run}" == true ]]; then
            session="mat_dry_${protocol}_${replicate_padded}"
        else
            session="mat_${protocol}_${replicate_padded}"
        fi
        [[ -z "${first_session}" ]] && first_session="${session}"
        logfile="${log_directory}/${session}.log"

        command_parts=(
            "${julia_binary}"
            "--startup-file=no"
            "--project=${project_root}"
            "${runner}"
            "--protocol" "${protocol}"
            "--replicate" "${replicate}"
            "--results-dir" "${results_directory}"
        )
        [[ "${worker_dry_run}" == true ]] && command_parts+=("--dry-run")
        [[ "${overwrite}" == true ]] && command_parts+=("--overwrite")
        if [[ "${use_systemd_inhibit}" == true ]]; then
            command_parts=(
                "${systemd_inhibit_binary}"
                "--what=${systemd_inhibit_what}"
                "--who=Oceananigans P3 ${session}"
                "--why=Paper revision Package 3 MAT stability worker"
                "--mode=block"
                "${command_parts[@]}"
            )
        fi

        printf -v worker_command "%q " "${command_parts[@]}"
        printf -v quoted_logfile "%q" "${logfile}"
        shell_command="set -o pipefail; ${worker_command}2>&1 | tee -a ${quoted_logfile}"
        printf -v quoted_shell_command "%q" "${shell_command}"
        tmux_command="bash -lc ${quoted_shell_command}"

        if tmux has-session -t "=${session}" 2>/dev/null; then
            echo "Skipping active tmux session ${session}"
            continue
        fi

        if [[ "${preview}" == true ]]; then
            echo "Would start ${session}: ${worker_command}"
        else
            tmux new-session -d -s "${session}" "${tmux_command}"
            echo "Started ${session}; log: ${logfile}"
        fi
    done
done

if [[ "${preview}" == false ]]; then
    echo
    echo "All requested workers were submitted."
    echo "Inspect sessions with: tmux ls"
    echo "Attach with:          tmux attach -t ${first_session}"
fi
