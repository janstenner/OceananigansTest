#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
julia_binary="${JULIA_BIN:-julia}"
results_directory="${MAT_EXPERT_RA1E5_RESULTS_DIR:-${script_directory}/results_ra1e5}"
run_count=10
master_seed="${MAT_EXPERT_RA1E5_MASTER_SEED:-20260901}"
episodes=""
threshold=""
preview=false
openblas_threads=3
omp_threads=1

usage() {
    cat <<'EOF'
Usage: Revision/MAT_expert_training/launch_tmux_ra1e5.sh [options]

Starts independent Varying-IC MAT training runs at Ra=1e5. Choose exactly one
stop rule:
  --episodes N              Every worker trains for exactly N episodes.
  --threshold VALUE         All workers stop at episode boundaries after the
                            first rolling-100 reward mean exceeds VALUE.

Options:
  --runs N                  Independent tmux training workers (default: 10).
  --master-seed N           Master seed for deterministic run/IC seeds.
  --results-dir PATH        Result root (default: results_ra1e5).
  --preview                 Validate and print commands; start nothing.
  --openblas-threads N      OpenBLAS threads per worker (default: 3).
  --omp-threads N           OpenMP threads per worker (default: 1).
  --help                    Show this message.

Rerunning the identical command resumes incomplete workers from latest.jld2.
Changing seeds, run count, stop rule, run file, or corpus requires a new result
directory because these identities are frozen in experiment_manifest.jld2.

Environment:
  JULIA_BIN                       Julia executable (default: julia)
  MAT_EXPERT_RA1E5_RESULTS_DIR    Result root
  MAT_EXPERT_RA1E5_MASTER_SEED    Master seed (default: 20260901)
EOF
}

while (($#)); do
    case "$1" in
        --episodes)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            episodes="$2"
            shift 2
            ;;
        --threshold)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            threshold="$2"
            shift 2
            ;;
        --runs)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            run_count="$2"
            shift 2
            ;;
        --master-seed|--master_seed)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            master_seed="$2"
            shift 2
            ;;
        --results-dir|--results_dir)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            results_directory="$2"
            shift 2
            ;;
        --preview)
            preview=true
            shift
            ;;
        --openblas-threads|--openblas_threads)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            openblas_threads="$2"
            shift 2
            ;;
        --omp-threads|--omp_threads)
            (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }
            omp_threads="$2"
            shift 2
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
done

[[ "${run_count}" =~ ^[1-9][0-9]*$ ]] || { echo "--runs must be positive." >&2; exit 2; }
[[ "${master_seed}" =~ ^[0-9]+$ ]] || { echo "--master-seed must be non-negative." >&2; exit 2; }
[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--openblas-threads must be positive." >&2; exit 2; }
[[ "${omp_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--omp-threads must be positive." >&2; exit 2; }
if [[ -n "${episodes}" && -n "${threshold}" ]] || [[ -z "${episodes}" && -z "${threshold}" ]]; then
    echo "Choose exactly one stop rule: --episodes N or --threshold VALUE." >&2
    exit 2
fi
if [[ -n "${episodes}" ]]; then
    [[ "${episodes}" =~ ^[1-9][0-9]*$ ]] || { echo "--episodes must be positive." >&2; exit 2; }
    stop_arguments=(--episodes "${episodes}")
else
    [[ "${threshold}" =~ ^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
        echo "--threshold must be a finite number." >&2
        exit 2
    }
    stop_arguments=(--threshold "${threshold}")
fi

export OPENBLAS_NUM_THREADS="${openblas_threads}"
export OMP_NUM_THREADS="${omp_threads}"

command -v "${julia_binary}" >/dev/null 2>&1 || {
    echo "Julia executable '${julia_binary}' was not found." >&2
    exit 1
}

worker_script="${script_directory}/run_ra1e5_training_worker.jl"
common_arguments=(
    --runs "${run_count}"
    --master-seed "${master_seed}"
    --results-dir "${results_directory}"
    "${stop_arguments[@]}"
)
prepare_command=(
    "${julia_binary}"
    "--startup-file=no"
    "--project=${project_root}"
    "${worker_script}"
    --prepare-only
    "${common_arguments[@]}"
)
[[ "${preview}" == true ]] && prepare_command+=(--preview)
"${prepare_command[@]}"

if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux was not found in PATH." >&2
        exit 1
    }
fi

launch_directory="${results_directory}/launches/$(date -u +%Y%m%dT%H%M%SZ)_$$"
if [[ "${preview}" == false ]]; then
    mkdir -p "${launch_directory}"
    printf 'session\trole\trun_index\tlog\n' > "${launch_directory}/jobs.tsv"
fi

build_command() {
    local -a command_parts=("$@")
    printf '%q ' "${command_parts[@]}"
}

start_session() {
    local session="$1"
    local run_index="$2"
    shift 2
    local logfile="${launch_directory}/${session}.log"
    local command
    command="$(build_command "$@")"
    if [[ "${preview}" == true ]]; then
        echo "Would start ${session}: ${command}"
        return
    fi
    if tmux has-session -t "=${session}" 2>/dev/null; then
        echo "Skipping active tmux session ${session}"
        return
    fi
    printf '%s\ttraining\t%s\t%s\n' "${session}" "${run_index}" "${logfile}" \
        >> "${launch_directory}/jobs.tsv"
    printf -v quoted_logfile '%q' "${logfile}"
    local shell_command="set -o pipefail; export OPENBLAS_NUM_THREADS=${openblas_threads}; export OMP_NUM_THREADS=${omp_threads}; ${command}2>&1 | tee -a ${quoted_logfile}"
    printf -v quoted_shell_command '%q' "${shell_command}"
    tmux new-session -d -s "${session}" "bash -lc ${quoted_shell_command}"
    echo "Started ${session}; log: ${logfile}"
}

width=${#run_count}
((width < 2)) && width=2
for ((index = 1; index <= run_count; index++)); do
    printf -v padded_index "%0${width}d" "${index}"
    session="pmat_ra1e5_${padded_index}"
    start_session \
        "${session}" "${index}" \
        "${julia_binary}" "--startup-file=no" "--project=${project_root}" \
        "${worker_script}" --run-index "${index}" "${common_arguments[@]}"
done

if [[ "${preview}" == true ]]; then
    echo "Preview complete: ${run_count} Ra=1e5 training sessions would be started."
else
    echo
    echo "Submitted ${run_count} Ra=1e5 training sessions."
    echo "Inspect sessions: tmux ls"
    echo "Attach:           tmux attach -t pmat_ra1e5_01"
    echo "Follow a log:     tail -f ${launch_directory}/pmat_ra1e5_01.log"
    echo "Manifest/logs:    ${launch_directory}"
fi
