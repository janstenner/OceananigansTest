#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
julia_binary="${JULIA_BIN:-julia}"
results_directory="${MAT_EXPERT_RESULTS_DIR:-${script_directory}/results}"
source_results_directory="${MAT_IPPO_RESULTS_DIR:-${script_directory}/../MAT_IPPO_Comparison/results}"
distillation_experts_directory="${DISTILLATION_EXPERTS_DIR:-${script_directory}/../Expert_Apprentice_Distillation/experts}"
preview=false

usage() {
    cat <<'EOF'
Usage: Revision/MAT_expert_training/launch_tmux.sh [options]

Starts exactly 21 detached sessions:
  pmat_fixed_01 through pmat_fixed_10
  pmat_varying_01 through pmat_varying_10
  pmat_test

The 20 training sessions continue all validation-ranked MAT checkpoints.
The test session waits for each protocol winner, then evaluates it on the
protocol test set, saves reward curves and scores, publishes compact
expert.jld2 files, and replaces both tracked Distillation experts.

Options:
  --results-dir PATH          Override expert-training result root.
  --source-results-dir PATH   Override MAT-IPPO Package-4 result root.
  --distillation-experts-dir PATH
                              Override tracked Distillation expert root.
  --preview                   Validate and print commands; start nothing.
  --help                      Show this message.

Environment:
  JULIA_BIN                  Julia executable (default: julia)
  MAT_EXPERT_RESULTS_DIR     Expert-training result root
  MAT_IPPO_RESULTS_DIR       MAT-IPPO Package-4 result root
  DISTILLATION_EXPERTS_DIR   Tracked Distillation expert root
EOF
}

while (($#)); do
    case "$1" in
        --results-dir|--results_dir)
            results_directory="$2"
            shift 2
            ;;
        --source-results-dir|--source_results_dir)
            source_results_directory="$2"
            shift 2
            ;;
        --distillation-experts-dir|--distillation_experts_dir)
            distillation_experts_directory="$2"
            shift 2
            ;;
        --preview)
            preview=true
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
done

command -v "${julia_binary}" >/dev/null 2>&1 || {
    echo "Julia executable '${julia_binary}' was not found." >&2
    exit 1
}

prepare_command=(
    "${julia_binary}"
    "--startup-file=no"
    "--project=${project_root}"
    "${script_directory}/prepare_experiment.jl"
    "--results-dir" "${results_directory}"
    "--source-results-dir" "${source_results_directory}"
)
[[ "${preview}" == true ]] && prepare_command+=("--preview")
"${prepare_command[@]}"

if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux was not found in PATH." >&2
        exit 1
    }
fi

declare -a fixed_run_ids=(
    seed_dfe17c7e95fcbb6d
    seed_92b79c49251eb7a2
    seed_14a5d7fe05cff1de
    seed_1f82af6b2a587ef6
    seed_7ebdabd4b2738e12
    seed_e72249f3ea1fa410
    seed_5954219d6cc69d5c
    seed_ce0b5b582dda8eff
    seed_4c43a7202fb90fbe
    seed_3a2d3a2f3341b412
)
declare -a varying_run_ids=(
    seed_5954219d6cc69d5c
    seed_e72249f3ea1fa410
    seed_14a5d7fe05cff1de
    seed_92b79c49251eb7a2
    seed_dfe17c7e95fcbb6d
    seed_1f82af6b2a587ef6
    seed_ce0b5b582dda8eff
    seed_3a2d3a2f3341b412
    seed_7ebdabd4b2738e12
    seed_4c43a7202fb90fbe
)

launch_directory="${results_directory}/launches/$(date -u +%Y%m%dT%H%M%SZ)_$$"
if [[ "${preview}" == false ]]; then
    mkdir -p "${launch_directory}"
    printf 'session\trole\tprotocol\trank\trun_id\tlog\n' > "${launch_directory}/jobs.tsv"
fi

build_command() {
    local -a command_parts=("$@")
    printf '%q ' "${command_parts[@]}"
}

start_session() {
    local session="$1"
    local role="$2"
    local protocol="$3"
    local rank="$4"
    local run_id="$5"
    shift 5
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
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${session}" "${role}" "${protocol}" "${rank}" "${run_id}" "${logfile}" \
        >> "${launch_directory}/jobs.tsv"
    printf -v quoted_logfile '%q' "${logfile}"
    local shell_command="set -o pipefail; ${command}2>&1 | tee -a ${quoted_logfile}"
    printf -v quoted_shell_command '%q' "${shell_command}"
    tmux new-session -d -s "${session}" "bash -lc ${quoted_shell_command}"
    echo "Started ${session}; log: ${logfile}"
}

for protocol in fixed varying; do
    if [[ "${protocol}" == fixed ]]; then
        run_ids=("${fixed_run_ids[@]}")
    else
        run_ids=("${varying_run_ids[@]}")
    fi
    for index in "${!run_ids[@]}"; do
        printf -v rank '%02d' "$((index + 1))"
        run_id="${run_ids[index]}"
        session="pmat_${protocol}_${rank}"
        start_session \
            "${session}" training "${protocol}" "${rank}" "${run_id}" \
            "${julia_binary}" "--startup-file=no" "--project=${project_root}" \
            "${script_directory}/run_training_worker.jl" \
            "--protocol" "${protocol}" \
            "--run-id" "${run_id}" \
            "--results-dir" "${results_directory}" \
            "--source-results-dir" "${source_results_directory}"
    done
done

start_session \
    pmat_test test all NA winners \
    "${julia_binary}" "--startup-file=no" "--project=${project_root}" \
    "${script_directory}/run_test_worker.jl" \
    "--results-dir" "${results_directory}" \
    "--distillation-experts-dir" "${distillation_experts_directory}"

if [[ "${preview}" == true ]]; then
    echo "Preview complete: exactly 21 sessions would be started."
else
    echo
    echo "Submitted 20 training sessions and one waiting test/export session."
    echo "Inspect sessions: tmux ls"
    echo "Attach:           tmux attach -t pmat_fixed_01"
    echo "Follow a log:     tail -f ${launch_directory}/pmat_fixed_01.log"
    echo "Manifest/logs:    ${launch_directory}"
fi
