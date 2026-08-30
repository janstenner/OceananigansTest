#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
worker="${script_directory}/run_baseline.jl"
results_directory="${BASELINE_RESULTS_DIR:-${script_directory}/results}"
julia_binary="${JULIA_BIN:-julia}"
fixed_expert="${script_directory}/../Expert_Apprentice_Distillation/experts/fixed/agent.jld2"
varying_expert="${script_directory}/../Expert_Apprentice_Distillation/experts/varying/agent.jld2"
protocol_selection=all
controller_selection=all
run_seed=600600
openblas_threads=3
omp_threads=1
preview=false
overwrite=false

usage() {
    cat <<'EOF'
Usage: bash Revision/Baselines/launch_tmux.sh [options]

  --protocol all|fixed|varying
  --controller all|expert|unactuated
  --results-dir PATH
  --fixed-expert-path PATH
  --varying-expert-path PATH
  --run-seed N
  --openblas-threads N     OpenBLAS threads per worker (default: 3).
  --omp-threads N          OpenMP threads per worker (default: 1).
  --overwrite
  --preview
  --help
EOF
}

while (($#)); do
    case "$1" in
        --protocol) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; protocol_selection="$2"; shift 2 ;;
        --controller) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; controller_selection="$2"; shift 2 ;;
        --results-dir) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; results_directory="$2"; shift 2 ;;
        --fixed-expert-path) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; fixed_expert="$2"; shift 2 ;;
        --varying-expert-path) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; varying_expert="$2"; shift 2 ;;
        --run-seed) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; run_seed="$2"; shift 2 ;;
        --openblas-threads) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; openblas_threads="$2"; shift 2 ;;
        --omp-threads) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; omp_threads="$2"; shift 2 ;;
        --overwrite) overwrite=true; shift ;;
        --preview) preview=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${protocol_selection}" == all || "${protocol_selection}" == fixed || "${protocol_selection}" == varying ]] || {
    echo "--protocol must be all, fixed, or varying." >&2; exit 2;
}
[[ "${controller_selection}" == all || "${controller_selection}" == expert || "${controller_selection}" == unactuated ]] || {
    echo "--controller must be all, expert, or unactuated." >&2; exit 2;
}
[[ "${run_seed}" =~ ^[0-9]+$ ]] || { echo "--run-seed must be nonnegative." >&2; exit 2; }
[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--openblas-threads must be positive." >&2; exit 2; }
[[ "${omp_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--omp-threads must be positive." >&2; exit 2; }
export OPENBLAS_NUM_THREADS="${openblas_threads}"
export OMP_NUM_THREADS="${omp_threads}"

command -v "${julia_binary}" >/dev/null 2>&1 || { echo "Julia executable not found: ${julia_binary}" >&2; exit 1; }
if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || { echo "tmux was not found in PATH." >&2; exit 1; }
fi

protocols=(fixed varying)
controllers=(expert unactuated)
[[ "${protocol_selection}" == all ]] || protocols=("${protocol_selection}")
[[ "${controller_selection}" == all ]] || controllers=("${controller_selection}")

if [[ "${preview}" == false ]]; then
    for protocol in "${protocols[@]}"; do
        for controller in "${controllers[@]}"; do
            session="baseline_${protocol}_${controller}"
            tmux has-session -t "${session}" 2>/dev/null && {
                echo "tmux session already exists: ${session}" >&2; exit 1;
            }
        done
    done
fi

launch_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
log_directory="${results_directory}/launches/${launch_id}"
[[ "${preview}" == true ]] || mkdir -p "${log_directory}"
printf -v quoted_project '%q' "${project_root}"
planned=0

for protocol in "${protocols[@]}"; do
    for controller in "${controllers[@]}"; do
        session="baseline_${protocol}_${controller}"
        logfile="${log_directory}/${protocol}_${controller}.log"
        command=(
            "${julia_binary}" --startup-file=no "--project=${project_root}" "${worker}"
            --protocol "${protocol}" --controller "${controller}"
            --results-dir "${results_directory}" --run-seed "${run_seed}"
        )
        if [[ "${controller}" == expert ]]; then
            expert_path="${fixed_expert}"
            [[ "${protocol}" == fixed ]] || expert_path="${varying_expert}"
            command+=(--expert-path "${expert_path}")
        fi
        [[ "${overwrite}" == false ]] || command+=(--overwrite)
        printf -v command_text '%q ' "${command[@]}"
        printf -v quoted_logfile '%q' "${logfile}"
        shell_command="set -o pipefail; cd ${quoted_project}; export OPENBLAS_NUM_THREADS=${openblas_threads}; export OMP_NUM_THREADS=${omp_threads}; ${command_text}2>&1 | tee -a ${quoted_logfile}"
        printf -v quoted_shell_command '%q' "${shell_command}"
        planned=$((planned + 1))
        if [[ "${preview}" == true ]]; then
            echo "${session}: bash -lc ${quoted_shell_command}"
        else
            tmux new-session -d -s "${session}" "bash -lc ${quoted_shell_command}"
            echo "Started ${session}; log: ${logfile}"
        fi
    done
done

if [[ "${preview}" == true ]]; then
    echo "Previewed ${planned} baseline workers."
else
    echo "Started ${planned} baseline workers. Results: ${results_directory}"
fi
