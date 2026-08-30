#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
training_worker="${script_directory}/run_training_worker.jl"
analysis_worker="${script_directory}/analyze_configuration_worker.jl"
manifest_worker="${script_directory}/prepare_manifest.jl"
results_directory="${P7_RESULTS_DIR:-${script_directory}/results}"
julia_binary="${JULIA_BIN:-julia}"

configuration_selection="all"
experiment_id=""
declare -a explicit_strengths=()
preview=false
analysis_only=false
retry_failed=false
planned=0
submitted=0
skipped=0
first_session=""
openblas_threads=3
omp_threads=1

usage() {
    cat <<'EOF'
Usage: launch_tmux.sh [options]

Without filters, starts all eight Package-7 configurations at their configured
three-strength grids: 72 training workers and eight analysis/wait workers.

Options:
  --config NAME         One of go-gc, go-sc, gr-gc, gr-sc,
                        group-lasso-gc, group-lasso-sc, growl-gc, growl-sc.
  --strength VALUE      Explicit strength for one selected configuration;
                        repeat to replace its configured strength grid.
  --experiment-id ID    Reuse a timestamp experiment directory. Automatically
                        generated for a new launch; required with --analysis-only.
  --preview             Print all planned sessions without writing or launching.
  --analysis-only       Start only the selected analysis/wait worker(s).
  --retry-failed        Permit training workers to resume failed runs.
  --results-dir PATH    Override the Package-7 result root.
  --openblas-threads N  OpenBLAS threads per worker (default: 3).
  --omp-threads N       OpenMP threads per worker (default: 1).
  --help                Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --config)
            (($# >= 2)) || { echo "Missing value after --config." >&2; exit 2; }
            configuration_selection="$2"
            shift
            ;;
        --experiment-id)
            (($# >= 2)) || { echo "Missing value after --experiment-id." >&2; exit 2; }
            experiment_id="$2"
            shift
            ;;
        --strength)
            (($# >= 2)) || { echo "Missing value after --strength." >&2; exit 2; }
            explicit_strengths+=("$2")
            shift
            ;;
        --results-dir)
            (($# >= 2)) || { echo "Missing value after --results-dir." >&2; exit 2; }
            results_directory="$2"
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
        --preview) preview=true ;;
        --analysis-only) analysis_only=true ;;
        --retry-failed) retry_failed=true ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--openblas-threads must be positive." >&2; exit 2; }
[[ "${omp_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--omp-threads must be positive." >&2; exit 2; }
export OPENBLAS_NUM_THREADS="${openblas_threads}"
export OMP_NUM_THREADS="${omp_threads}"

[[ "${analysis_only}" == false || -n "${experiment_id}" ]] || {
    echo "--analysis-only requires --experiment-id so the existing runs can be located." >&2
    exit 2
}
[[ -n "${experiment_id}" ]] || experiment_id="$(date -u +%y%m%d_%H%M%S)"
[[ "${experiment_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
    echo "Invalid --experiment-id '${experiment_id}'. Use only letters, digits, underscores, and hyphens." >&2
    exit 2
}

command -v "${julia_binary}" >/dev/null 2>&1 || { echo "Julia executable '${julia_binary}' was not found." >&2; exit 1; }
if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || { echo "tmux was not found in PATH." >&2; exit 1; }
fi

variant_arguments=(--print-variants --config "${configuration_selection}")
for strength in "${explicit_strengths[@]}"; do
    variant_arguments+=(--strength "${strength}")
done
variant_output="$("${julia_binary}" --startup-file=no "--project=${project_root}" "${manifest_worker}" "${variant_arguments[@]}")"
mapfile -t variant_rows <<< "${variant_output}"
((${#variant_rows[@]} > 0)) || { echo "No Package-7 variants selected." >&2; exit 2; }

declare -A strengths_by_configuration=()
selected_configurations=()
for row in "${variant_rows[@]}"; do
    IFS=$'\t' read -r configuration strength <<< "${row}"
    if [[ -z "${strengths_by_configuration[${configuration}]+x}" ]]; then
        selected_configurations+=("${configuration}")
        strengths_by_configuration[${configuration}]="${strength}"
    else
        strengths_by_configuration[${configuration}]+=" ${strength}"
    fi
done

launch_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
launch_directory="${results_directory}/${experiment_id}/launches/${launch_id}"
log_directory="${launch_directory}/logs"
job_manifest="${launch_directory}/jobs.tsv"

manifest_arguments=(--output "${launch_directory}/study_manifest.jld2" --experiment-id "${experiment_id}" --config "${configuration_selection}" --results-dir "${results_directory}")
for strength in "${explicit_strengths[@]}"; do
    manifest_arguments+=(--strength "${strength}")
done

if [[ "${preview}" == false ]]; then
    mkdir -p "${log_directory}"
    printf 'session\tkind\tconfiguration\tstrength\treplicate\tdisposition\tlog\n' > "${job_manifest}.tmp"
    "${julia_binary}" --startup-file=no "--project=${project_root}" "${manifest_worker}" \
        "${manifest_arguments[@]}" > "${launch_directory}/manifest.log" 2>&1
fi

safe_tag() {
    local value="${1,,}"
    value="${value//./p}"
    value="${value//-/m}"
    value="${value//+/}"
    echo "${value}"
}

start_session() {
    local session="$1"
    local kind="$2"
    local configuration="$3"
    local strength="$4"
    local replicate="$5"
    shift 5
    local logfile="${log_directory}/${session}.log"
    local worker_command quoted_logfile shell_command quoted_shell_command tmux_command
    local -a command_parts=("$@")

    planned=$((planned + 1))
    [[ -n "${first_session}" ]] || first_session="${session}"
    printf -v worker_command "%q " "${command_parts[@]}"
    if [[ "${preview}" == true ]]; then
        echo "Would start ${session}: ${worker_command}"
        return
    fi
    if tmux has-session -t "=${session}" 2>/dev/null; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${session}" "${kind}" "${configuration}" "${strength}" "${replicate}" skipped_active "${logfile}" >> "${job_manifest}.tmp"
        echo "Skipping active tmux session ${session}"
        skipped=$((skipped + 1))
        return
    fi
    printf -v quoted_logfile "%q" "${logfile}"
    shell_command="set -o pipefail; export OPENBLAS_NUM_THREADS=${openblas_threads}; export OMP_NUM_THREADS=${omp_threads}; ${worker_command}2>&1 | tee -a ${quoted_logfile}"
    printf -v quoted_shell_command "%q" "${shell_command}"
    tmux_command="bash -lc ${quoted_shell_command}"
    tmux new-session -d -s "${session}" "${tmux_command}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${session}" "${kind}" "${configuration}" "${strength}" "${replicate}" submitted "${logfile}" >> "${job_manifest}.tmp"
    echo "Started ${session}; log: ${logfile}"
    submitted=$((submitted + 1))
}

for configuration in "${selected_configurations[@]}"; do
    read -r -a strengths <<< "${strengths_by_configuration[${configuration}]}"
    configuration_tag="${configuration//-/_}"
    for strength in "${strengths[@]}"; do
        strength_session_tag="$(safe_tag "${strength}")"
        if [[ "${analysis_only}" == false ]]; then
            for replicate in 1 2 3; do
                replicate_tag="$(printf 'r%02d' "${replicate}")"
                session="p7_${experiment_id}_${configuration_tag}_${strength_session_tag}_${replicate_tag}"
                command=(
                    "${julia_binary}" --startup-file=no "--project=${project_root}" "${training_worker}"
                    --experiment-id "${experiment_id}" --config "${configuration}" --strength "${strength}"
                    --replicate "${replicate}" --results-dir "${results_directory}"
                )
                [[ "${retry_failed}" == false ]] || command+=(--retry-failed)
                start_session "${session}" training "${configuration}" "${strength}" "${replicate}" "${command[@]}"
            done
        fi
    done
    session="p7_${experiment_id}_${configuration_tag}_analyze"
    analysis_command=(
        "${julia_binary}" --startup-file=no "--project=${project_root}" "${analysis_worker}"
        --experiment-id "${experiment_id}" --config "${configuration}" --results-dir "${results_directory}"
    )
    for strength in "${strengths[@]}"; do
        analysis_command+=(--strength "${strength}")
    done
    [[ "${retry_failed}" == false ]] || analysis_command+=(--retry-failed)
    start_session "${session}" analysis "${configuration}" "${strengths[*]}" 0 "${analysis_command[@]}"
done

if [[ "${preview}" == true ]]; then
    echo
    echo "Previewed ${planned} sessions. No files or processes were created."
    echo "Experiment ID: ${experiment_id}"
    exit 0
fi

mv "${job_manifest}.tmp" "${job_manifest}"
{
    echo "launch_id=${launch_id}"
    echo "experiment_id=${experiment_id}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "configuration=${configuration_selection}"
    echo "explicit_strengths=${explicit_strengths[*]}"
    echo "analysis_only=${analysis_only}"
    echo "retry_failed=${retry_failed}"
    echo "openblas_threads=${openblas_threads}"
    echo "omp_threads=${omp_threads}"
    echo "planned_sessions=${planned}"
    echo "submitted_sessions=${submitted}"
    echo "skipped_active_sessions=${skipped}"
    echo "results_directory=${results_directory}"
} > "${launch_directory}/launch.env"

echo
echo "Planned ${planned} sessions; submitted ${submitted}; skipped ${skipped} active sessions."
echo "Experiment ID: ${experiment_id}"
echo "Result directory: ${results_directory}/${experiment_id}"
echo "All sessions close automatically when their worker exits."
echo "Launch manifest: ${job_manifest}"
echo "Inspect sessions: tmux ls"
[[ -z "${first_session}" ]] || echo "Attach: tmux attach -t ${first_session}"
