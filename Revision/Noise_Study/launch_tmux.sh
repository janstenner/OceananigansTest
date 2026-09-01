#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
manifest_worker="${script_directory}/prepare_manifest.jl"
noise_worker="${script_directory}/run_worker.jl"
results_directory="${NOISE_STUDY_RESULTS_DIR:-${script_directory}/results}"
julia_binary="${JULIA_BIN:-julia}"

protocol_selection=all
controller_selection=all
experiment_id=""
declare -a explicit_levels=()
package7_results="${script_directory}/../Package7/results/260830_173924"
package8_results="${script_directory}/../Package8/results/260830_231109"
go_study_results="${script_directory}/../GO_Sensitivity/results/study"
baseline_results="${script_directory}/../Baselines/results"
distillation_root="${script_directory}/../Expert_Apprentice_Distillation"
preview=false
retry_failed=false
rebuild_manifest=false
openblas_threads=3
omp_threads=1
planned=0
submitted=0
skipped=0
first_session=""

usage() {
    cat <<'EOF'
Usage: bash Revision/Noise_Study/launch_tmux.sh [options]

Without filters, creates/reuses one frozen manifest per protocol and starts 30
self-closing tmux sessions: three controllers times five noise levels times two
protocols. One session owns all cases and replicates for its combination.

  --protocol all|fixed|varying
  --controller all|expert|sparse|c_match
  --noise-level LEVEL       Repeat to select from 0, 0.01, 0.05, 0.10, 0.20.
  --experiment-id ID        Reuse an existing experiment; generated otherwise.
  --results-dir PATH
  --package7-results PATH
  --package8-results PATH
  --go-study-results PATH
  --baseline-results PATH
  --distillation-root PATH
  --openblas-threads N      OpenBLAS threads per worker (default: 3).
  --omp-threads N           OpenMP threads per worker (default: 1).
  --retry-failed
  --rebuild-manifest        Recompute scales and replace selected manifests.
  --preview                 Print sessions without writing or launching.
  --help

Manifest creation computes exact protocol-specific channel scales from the
complete training corpus and can therefore take several minutes for Varying IC.
EOF
}

while (($#)); do
    case "$1" in
        --protocol) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; protocol_selection="$2"; shift 2 ;;
        --controller) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; controller_selection="$2"; shift 2 ;;
        --noise-level) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; explicit_levels+=("$2"); shift 2 ;;
        --experiment-id) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; experiment_id="$2"; shift 2 ;;
        --results-dir) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; results_directory="$2"; shift 2 ;;
        --package7-results) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; package7_results="$2"; shift 2 ;;
        --package8-results) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; package8_results="$2"; shift 2 ;;
        --go-study-results) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; go_study_results="$2"; shift 2 ;;
        --baseline-results) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; baseline_results="$2"; shift 2 ;;
        --distillation-root) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; distillation_root="$2"; shift 2 ;;
        --openblas-threads) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; openblas_threads="$2"; shift 2 ;;
        --omp-threads) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; omp_threads="$2"; shift 2 ;;
        --retry-failed) retry_failed=true; shift ;;
        --rebuild-manifest) rebuild_manifest=true; shift ;;
        --preview) preview=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${protocol_selection}" == all || "${protocol_selection}" == fixed || "${protocol_selection}" == varying ]] || {
    echo "--protocol must be all, fixed, or varying." >&2; exit 2;
}
[[ "${controller_selection}" == all || "${controller_selection}" == expert || "${controller_selection}" == sparse || "${controller_selection}" == c_match ]] || {
    echo "--controller must be all, expert, sparse, or c_match." >&2; exit 2;
}
[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--openblas-threads must be positive." >&2; exit 2; }
[[ "${omp_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--omp-threads must be positive." >&2; exit 2; }
[[ -n "${experiment_id}" ]] || experiment_id="$(date -u +%y%m%d_%H%M%S)"
[[ "${experiment_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo "Invalid experiment ID '${experiment_id}'." >&2; exit 2; }

protocols=(fixed varying)
controllers=(expert sparse c_match)
levels=(0 0.01 0.05 0.10 0.20)
[[ "${protocol_selection}" == all ]] || protocols=("${protocol_selection}")
[[ "${controller_selection}" == all ]] || controllers=("${controller_selection}")
if ((${#explicit_levels[@]})); then
    levels=()
    for requested in "${explicit_levels[@]}"; do
        case "${requested}" in
            0|0.0|0.00) normalized=0 ;;
            0.01) normalized=0.01 ;;
            0.05) normalized=0.05 ;;
            0.1|0.10) normalized=0.10 ;;
            0.2|0.20) normalized=0.20 ;;
            *) echo "Unsupported --noise-level '${requested}'." >&2; exit 2 ;;
        esac
        [[ " ${levels[*]} " == *" ${normalized} "* ]] || levels+=("${normalized}")
    done
fi

command -v "${julia_binary}" >/dev/null 2>&1 || { echo "Julia executable not found: ${julia_binary}" >&2; exit 1; }
if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || { echo "tmux was not found in PATH." >&2; exit 1; }
fi

level_tag() {
    case "$1" in
        0) echo n_0p00 ;;
        0.01) echo n_0p01 ;;
        0.05) echo n_0p05 ;;
        0.10) echo n_0p10 ;;
        0.20) echo n_0p20 ;;
        *) return 2 ;;
    esac
}

experiment_directory="${results_directory}/${experiment_id}"
launch_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
launch_directory="${experiment_directory}/launches/${launch_id}"
log_directory="${launch_directory}/logs"
job_manifest="${launch_directory}/jobs.tsv"

if [[ "${preview}" == false ]]; then
    mkdir -p "${log_directory}" "${experiment_directory}/manifests"
    printf 'session\tprotocol\tcontroller\tnoise_level\tdisposition\tlog\n' > "${job_manifest}.tmp"
fi

for protocol in "${protocols[@]}"; do
    manifest="${experiment_directory}/manifests/${protocol}.jld2"
    if [[ "${preview}" == true ]]; then
        echo "Would prepare/reuse manifest: ${manifest}"
    elif [[ ! -f "${manifest}" || "${rebuild_manifest}" == true ]]; then
        package_results="${package7_results}"
        [[ "${protocol}" == fixed ]] || package_results="${package8_results}"
        manifest_log="${launch_directory}/manifest_${protocol}.log"
        echo "Preparing ${protocol} manifest; log: ${manifest_log}"
        "${julia_binary}" --startup-file=no "--project=${project_root}" "${manifest_worker}" \
            --protocol "${protocol}" --experiment-id "${experiment_id}" --output "${manifest}" \
            --package-results "${package_results}" --go-study-results "${go_study_results}" \
            --baseline-results "${baseline_results}" --distillation-root "${distillation_root}" \
            > "${manifest_log}" 2>&1
    else
        echo "Reusing frozen manifest: ${manifest}"
    fi

    for level in "${levels[@]}"; do
        tag="$(level_tag "${level}")"
        for controller in "${controllers[@]}"; do
            session="p10_${experiment_id}_${protocol}_${controller}_${tag}"
            session="${session//-/_}"
            logfile="${log_directory}/${session}.log"
            command=(
                "${julia_binary}" --startup-file=no "--project=${project_root}" "${noise_worker}"
                --manifest "${manifest}" --protocol "${protocol}" --controller "${controller}"
                --noise-level "${level}" --results-dir "${results_directory}"
            )
            [[ "${retry_failed}" == false ]] || command+=(--retry-failed)
            printf -v command_text '%q ' "${command[@]}"
            planned=$((planned + 1))
            [[ -n "${first_session}" ]] || first_session="${session}"
            if [[ "${preview}" == true ]]; then
                echo "Would start ${session}: ${command_text}"
                continue
            fi
            if tmux has-session -t "=${session}" 2>/dev/null; then
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${session}" "${protocol}" "${controller}" "${level}" skipped_active "${logfile}" >> "${job_manifest}.tmp"
                echo "Skipping active tmux session ${session}"
                skipped=$((skipped + 1))
                continue
            fi
            printf -v quoted_logfile '%q' "${logfile}"
            shell_command="set -o pipefail; export OPENBLAS_NUM_THREADS=${openblas_threads}; export OMP_NUM_THREADS=${omp_threads}; ${command_text}2>&1 | tee -a ${quoted_logfile}"
            printf -v quoted_shell_command '%q' "${shell_command}"
            tmux new-session -d -s "${session}" "bash -lc ${quoted_shell_command}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${session}" "${protocol}" "${controller}" "${level}" submitted "${logfile}" >> "${job_manifest}.tmp"
            echo "Started ${session}; log: ${logfile}"
            submitted=$((submitted + 1))
        done
    done
done

if [[ "${preview}" == true ]]; then
    echo
    echo "Previewed ${planned} Noise-Study workers. No files or processes were created."
    echo "Experiment ID: ${experiment_id}"
    exit 0
fi

mv "${job_manifest}.tmp" "${job_manifest}"
{
    echo "launch_id=${launch_id}"
    echo "experiment_id=${experiment_id}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "protocol=${protocol_selection}"
    echo "controller=${controller_selection}"
    echo "noise_levels=${levels[*]}"
    echo "retry_failed=${retry_failed}"
    echo "rebuild_manifest=${rebuild_manifest}"
    echo "openblas_threads=${openblas_threads}"
    echo "omp_threads=${omp_threads}"
    echo "planned_sessions=${planned}"
    echo "submitted_sessions=${submitted}"
    echo "skipped_active_sessions=${skipped}"
    echo "results_directory=${results_directory}"
} > "${launch_directory}/launch.env"

echo
echo "Planned ${planned} sessions; submitted ${submitted}; skipped ${skipped} active sessions."
echo "Experiment directory: ${experiment_directory}"
echo "Launch manifest: ${job_manifest}"
echo "All sessions close automatically when their worker exits."
echo "Inspect sessions: tmux ls"
[[ -z "${first_session}" ]] || echo "Attach: tmux attach -t ${first_session}"
