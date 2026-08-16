#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
training_worker="${script_directory}/run_study_worker.jl"
analysis_worker="${script_directory}/analyze_study_worker.jl"
manifest_worker="${script_directory}/prepare_study_manifest.jl"
results_directory="${P6_STUDY_RESULTS_DIR:-${script_directory}/results/study}"
julia_binary="${JULIA_BIN:-julia}"

preview=false
analysis_only=false
parallel_test=false
protocol_selection=all
submitted=0
skipped=0
planned=0
first_session=""

usage() {
    cat <<'EOF'
Usage: launch_study_tmux.sh [options]

Starts all selected Package-6 SC study jobs as independent, detached and
self-closing tmux sessions. The default starts 36 training sessions and two
analysis/wait sessions concurrently.

Options:
  --protocol VALUE       Select all, fixed, or varying (default: all).
  --preview              Print planned sessions without writing or launching.
  --analysis-only        Start only the selected analysis/wait session(s).
  --parallel-test        Run every terminal test episode in its own process.
  --results-dir PATH     Override the result root.
  --help                 Show this help.

The analysis workers poll every 60 seconds for at most 14 days and terminate
with a failure report if any training worker is explicitly failed.
EOF
}

while (($#)); do
    case "$1" in
        --preview) preview=true ;;
        --analysis-only) analysis_only=true ;;
        --parallel-test) parallel_test=true ;;
        --protocol)
            (($# >= 2)) || { echo "Missing value after --protocol." >&2; exit 2; }
            protocol_selection="$2"
            shift
            ;;
        --results-dir)
            (($# >= 2)) || { echo "Missing value after --results-dir." >&2; exit 2; }
            results_directory="$2"
            shift
            ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "${protocol_selection}" in
    all) protocols=(fixed varying) ;;
    fixed) protocols=(fixed) ;;
    varying) protocols=(varying) ;;
    *) echo "Invalid protocol '${protocol_selection}'; use all, fixed, or varying." >&2; exit 2 ;;
esac

if [[ "${preview}" == false ]]; then
    command -v tmux >/dev/null 2>&1 || { echo "tmux was not found in PATH." >&2; exit 1; }
    command -v "${julia_binary}" >/dev/null 2>&1 || { echo "Julia executable '${julia_binary}' was not found." >&2; exit 1; }
fi

launch_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
launch_directory="${results_directory}/launches/${launch_id}"
log_directory="${launch_directory}/logs"
job_manifest="${launch_directory}/jobs.tsv"

if [[ "${preview}" == false ]]; then
    mkdir -p "${log_directory}"
    printf 'session\tkind\tprotocol\tmethod\tstrength_index\treplicate\tdisposition\tlog\n' > "${job_manifest}.tmp"
    "${julia_binary}" --startup-file=no "--project=${project_root}" "${manifest_worker}" \
        --output "${launch_directory}/study_manifest.jld2" \
        --protocol "${protocol_selection}" --results-dir "${results_directory}" \
        > "${launch_directory}/manifest.log" 2>&1
fi

start_session() {
    local session="$1"
    local kind="$2"
    local protocol="$3"
    local method="$4"
    local strength_index="$5"
    local replicate="$6"
    shift 6
    local logfile="${log_directory}/${session}.log"
    local worker_command
    local quoted_logfile
    local shell_command
    local quoted_shell_command
    local tmux_command
    local -a command_parts=("$@")

    planned=$((planned + 1))
    [[ -n "${first_session}" ]] || first_session="${session}"
    printf -v worker_command "%q " "${command_parts[@]}"
    if [[ "${preview}" == true ]]; then
        echo "Would start ${session}: ${worker_command}"
        return
    fi
    if tmux has-session -t "=${session}" 2>/dev/null; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${session}" "${kind}" "${protocol}" "${method}" "${strength_index}" "${replicate}" skipped_active "${logfile}" >> "${job_manifest}.tmp"
        echo "Skipping active tmux session ${session}"
        skipped=$((skipped + 1))
        return
    fi
    printf -v quoted_logfile "%q" "${logfile}"
    shell_command="set -o pipefail; ${worker_command}2>&1 | tee -a ${quoted_logfile}"
    printf -v quoted_shell_command "%q" "${shell_command}"
    tmux_command="bash -lc ${quoted_shell_command}"
    tmux new-session -d -s "${session}" "${tmux_command}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${session}" "${kind}" "${protocol}" "${method}" "${strength_index}" "${replicate}" submitted "${logfile}" >> "${job_manifest}.tmp"
    echo "Started ${session}; log: ${logfile}"
    submitted=$((submitted + 1))
}

for protocol in "${protocols[@]}"; do
    protocol_tag=f
    [[ "${protocol}" == varying ]] && protocol_tag=v
    if [[ "${analysis_only}" == false ]]; then
        for strength_index in 1 2 3 4 5; do
            strength_tag="$(printf 's%02d' "${strength_index}")"
            for replicate in 1 2 3; do
                replicate_tag="$(printf 'r%02d' "${replicate}")"
                session="p6_${protocol_tag}_go_${strength_tag}_${replicate_tag}"
                start_session "${session}" training "${protocol}" go "${strength_index}" "${replicate}" \
                    "${julia_binary}" "--startup-file=no" "--project=${project_root}" "${training_worker}" \
                    --protocol "${protocol}" --method go --strength-index "${strength_index}" \
                    --replicate "${replicate}" --results-dir "${results_directory}"
            done
        done
        for replicate in 1 2 3; do
            replicate_tag="$(printf 'r%02d' "${replicate}")"
            session="p6_${protocol_tag}_gr_${replicate_tag}"
            start_session "${session}" training "${protocol}" gr 0 "${replicate}" \
                "${julia_binary}" "--startup-file=no" "--project=${project_root}" "${training_worker}" \
                --protocol "${protocol}" --method gr --strength-index 0 \
                --replicate "${replicate}" --results-dir "${results_directory}"
        done
    fi
    session="p6_${protocol_tag}_analyze"
    analysis_command=(
        "${julia_binary}" "--startup-file=no" "--project=${project_root}" "${analysis_worker}"
        --protocol "${protocol}" --results-dir "${results_directory}"
    )
    [[ "${parallel_test}" == false ]] || analysis_command+=(--parallel-test)
    start_session "${session}" analysis "${protocol}" none 0 0 \
        "${analysis_command[@]}"
done

if [[ "${preview}" == true ]]; then
    echo
    echo "Previewed ${planned} sessions. No files or processes were created."
    exit 0
fi

mv "${job_manifest}.tmp" "${job_manifest}"
{
    echo "launch_id=${launch_id}"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "protocol=${protocol_selection}"
    echo "analysis_only=${analysis_only}"
    echo "parallel_test=${parallel_test}"
    echo "planned_sessions=${planned}"
    echo "submitted_sessions=${submitted}"
    echo "skipped_active_sessions=${skipped}"
    echo "results_directory=${results_directory}"
} > "${launch_directory}/launch.env"

echo
echo "Planned ${planned} sessions; submitted ${submitted}; skipped ${skipped} active sessions."
echo "All sessions close automatically when their process exits."
echo "Launch manifest: ${job_manifest}"
echo "Inspect sessions: tmux ls"
[[ -z "${first_session}" ]] || echo "Attach: tmux attach -t ${first_session}"
