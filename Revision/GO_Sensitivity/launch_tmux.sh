#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
worker="${script_directory}/run_strength_calibration_worker.jl"
results_directory="${P6_CALIBRATION_RESULTS_DIR:-${script_directory}/results/strength_calibration}"
julia_binary="${JULIA_BIN:-julia}"

preview=false
protocol_selection=all
grouping_selection=all
submitted=0
skipped=0
first_session=""
openblas_threads=3
omp_threads=1

fixed_grouped_strengths=(0.003 0.006 0.01 0.03 0.06 0.09)
fixed_separate_strengths=(0.0015 0.003 0.006 0.01 0.02 0.03)
varying_grouped_strengths=(0.003 0.008 0.025 0.04 0.06)
varying_separate_strengths=(0.003 0.008 0.025 0.04 0.06)

usage() {
    cat <<'EOF'
Usage: launch_tmux.sh [options]

Starts one detached, self-closing tmux session for every selected Package-6
strength-calibration case. With the default selections this starts all 22
workers in parallel: six Fixed-GC, six Fixed-SC, five Varying-GC, and five
Varying-SC workers.

Options:
  --protocol VALUE   Select all, fixed, or varying workers (default: all).
  --grouping VALUE   Select all, grouped, or separate workers (default: all).
  --preview          Print the 22 planned sessions without starting them.
  --openblas-threads N  OpenBLAS threads per worker (default: 3).
  --omp-threads N       OpenMP threads per worker (default: 1).
  --help             Show this message.

Environment:
  JULIA_BIN                   Julia executable (default: julia)
  P6_CALIBRATION_RESULTS_DIR  Strength-calibration result root
Every worker uses regression learning rate 2e-4. Fixed workers run for 35,000
updates and Varying workers for 50,000 updates. Complete runs are skipped and
interrupted runs resume from their latest saved checkpoint.
EOF
}

while (($#)); do
    case "$1" in
        --preview)
            preview=true
            ;;
        --protocol)
            (($# >= 2)) || {
                echo "Missing value after --protocol." >&2
                exit 2
            }
            protocol_selection="$2"
            shift
            ;;
        --grouping)
            (($# >= 2)) || {
                echo "Missing value after --grouping." >&2
                exit 2
            }
            grouping_selection="$2"
            shift
            ;;
        --openblas-threads)
            (($# >= 2)) || { echo "Missing value after --openblas-threads." >&2; exit 2; }
            openblas_threads="$2"
            shift
            ;;
        --omp-threads)
            (($# >= 2)) || { echo "Missing value after --omp-threads." >&2; exit 2; }
            omp_threads="$2"
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

[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--openblas-threads must be positive." >&2; exit 2; }
[[ "${omp_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--omp-threads must be positive." >&2; exit 2; }
export OPENBLAS_NUM_THREADS="${openblas_threads}"
export OMP_NUM_THREADS="${omp_threads}"

case "${protocol_selection}" in
    all) protocols=(fixed varying) ;;
    fixed) protocols=(fixed) ;;
    varying) protocols=(varying) ;;
    *)
        echo "Invalid protocol '${protocol_selection}'; use all, fixed, or varying." >&2
        exit 2
        ;;
esac

case "${grouping_selection}" in
    all) groupings=(grouped separate) ;;
    grouped) groupings=(grouped) ;;
    separate) groupings=(separate) ;;
    *)
        echo "Invalid grouping '${grouping_selection}'; use all, grouped, or separate." >&2
        exit 2
        ;;
esac

if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux was not found in PATH." >&2
        exit 1
    }
    command -v "${julia_binary}" >/dev/null 2>&1 || {
        echo "Julia executable '${julia_binary}' was not found." >&2
        exit 1
    }
fi

log_directory="${results_directory}/logs"
[[ "${preview}" == true ]] || mkdir -p "${log_directory}"

start_worker() {
    local protocol="$1"
    local grouping="$2"
    local group_channels="$3"
    local strength="$4"
    local strength_tag="${strength//./p}"
    local grouping_tag
    local session
    local logfile
    local worker_command
    local quoted_logfile
    local shell_command
    local quoted_shell_command
    local tmux_command
    local -a command_parts

    if [[ "${grouping}" == grouped ]]; then
        grouping_tag=gc
    else
        grouping_tag=sc
    fi
    session="p6_cal_${protocol}_${grouping_tag}_s${strength_tag}"
    logfile="${log_directory}/${session}.log"
    [[ -n "${first_session}" ]] || first_session="${session}"

    command_parts=(
        "${julia_binary}"
        "--startup-file=no"
        "--project=${project_root}"
        "${worker}"
        "--protocol" "${protocol}"
        "--group-channels" "${group_channels}"
        "--strength" "${strength}"
        "--results-dir" "${results_directory}"
    )
    printf -v worker_command "%q " "${command_parts[@]}"
    if [[ "${preview}" == true ]]; then
        echo "Would start ${session}: ${worker_command}"
        submitted=$((submitted + 1))
        return
    fi

    if tmux has-session -t "=${session}" 2>/dev/null; then
        echo "Skipping active tmux session ${session}"
        skipped=$((skipped + 1))
        return
    fi

    printf -v quoted_logfile "%q" "${logfile}"
    shell_command="set -o pipefail; export OPENBLAS_NUM_THREADS=${openblas_threads}; export OMP_NUM_THREADS=${omp_threads}; ${worker_command}2>&1 | tee -a ${quoted_logfile}"
    printf -v quoted_shell_command "%q" "${shell_command}"
    tmux_command="bash -lc ${quoted_shell_command}"
    tmux new-session -d -s "${session}" "${tmux_command}"
    echo "Started ${session}; log: ${logfile}"
    submitted=$((submitted + 1))
}

for protocol in "${protocols[@]}"; do
    for grouping in "${groupings[@]}"; do
        if [[ "${protocol}_${grouping}" == fixed_grouped ]]; then
            strengths=("${fixed_grouped_strengths[@]}")
            group_channels=true
        elif [[ "${protocol}_${grouping}" == fixed_separate ]]; then
            strengths=("${fixed_separate_strengths[@]}")
            group_channels=false
        elif [[ "${protocol}_${grouping}" == varying_grouped ]]; then
            strengths=("${varying_grouped_strengths[@]}")
            group_channels=true
        else
            strengths=("${varying_separate_strengths[@]}")
            group_channels=false
        fi
        for strength in "${strengths[@]}"; do
            start_worker "${protocol}" "${grouping}" "${group_channels}" "${strength}"
        done
    done
done

if [[ "${preview}" == true ]]; then
    echo
    echo "Previewed ${submitted} worker sessions. No process was started."
else
    echo
    echo "Submitted ${submitted} worker sessions; skipped ${skipped} active sessions."
    echo "Each session closes automatically when its worker exits."
    echo "Inspect sessions with: tmux ls"
    [[ -z "${first_session}" ]] || echo "Attach with:          tmux attach -t ${first_session}"
    echo "Inspect logs with:    tail -f ${log_directory}/<session>.log"
fi
