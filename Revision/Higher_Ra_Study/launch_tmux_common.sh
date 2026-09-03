#!/usr/bin/env bash
set -euo pipefail

(($# >= 1)) || { echo "Internal error: missing Higher-Ra study." >&2; exit 2; }
study="$1"
shift
[[ "${study}" == ra5e4 || "${study}" == ra1e5 ]] || {
    echo "Internal error: study must be ra5e4 or ra1e5." >&2; exit 2;
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_directory}/../.." && pwd)"
julia_binary="${JULIA_BIN:-julia}"
split=train
max_workers=40
worker_directory="${script_directory}/Distillation_Corpuses/${study}/worker_results"
expert_path="${script_directory}/experts/${study}/expert.jld2"
run_seed=600600
preview=false
overwrite=false
openblas_threads=3
omp_threads=1

usage() {
    cat <<EOF
Usage: Revision/Higher_Ra_Study/launch_tmux_${study}.sh [options]

Creates the ${study} expert-rollout distillation corpus. The default launch is
the 40-worker training split: 20 basis snapshots times two mirrors. Each
training worker evaluates all 96 offsets. Validation and test workers evaluate
offsets 0 and 20.

  --split all|train|validation|test  Default: train.
  --max-workers N                    Concurrent persistent tmux slots; default: 40.
  --worker-dir PATH                  Result root for this distillation corpus.
  --expert-path PATH                 Matching compact expert.
  --run-seed N                       Runtime initialization seed; default: 600600.
  --openblas-threads N               OpenBLAS threads per worker; default: 3.
  --omp-threads N                    OpenMP threads per worker; default: 1.
  --preview                          Print jobs and commands; start nothing.
  --overwrite                        Regenerate matching complete shards.
  --help
EOF
}

while (($#)); do
    case "$1" in
        --split) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; split="$2"; shift 2 ;;
        --max-workers|--max_workers) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; max_workers="$2"; shift 2 ;;
        --worker-dir|--worker_dir) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; worker_directory="$2"; shift 2 ;;
        --expert-path|--expert_path) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; expert_path="$2"; shift 2 ;;
        --run-seed|--run_seed) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; run_seed="$2"; shift 2 ;;
        --openblas-threads|--openblas_threads) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; openblas_threads="$2"; shift 2 ;;
        --omp-threads|--omp_threads) (($# >= 2)) || { echo "Missing value after $1." >&2; exit 2; }; omp_threads="$2"; shift 2 ;;
        --preview) preview=true; shift ;;
        --overwrite) overwrite=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${split}" =~ ^(all|train|validation|test)$ ]] || { echo "Invalid --split." >&2; exit 2; }
[[ "${max_workers}" =~ ^[1-9][0-9]*$ ]] || { echo "--max-workers must be positive." >&2; exit 2; }
[[ "${run_seed}" =~ ^[0-9]+$ ]] || { echo "--run-seed must be nonnegative." >&2; exit 2; }
[[ "${openblas_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--openblas-threads must be positive." >&2; exit 2; }
[[ "${omp_threads}" =~ ^[1-9][0-9]*$ ]] || { echo "--omp-threads must be positive." >&2; exit 2; }
command -v "${julia_binary}" >/dev/null 2>&1 || { echo "Julia executable not found: ${julia_binary}" >&2; exit 1; }

if [[ "${preview}" == true ]]; then
    jobs_file="$(mktemp)"
    trap 'rm -f "${jobs_file}"' EXIT
    launch_directory=""
else
    launch_id="$(date -u +%Y%m%dT%H%M%SZ)_$$"
    launch_directory="${worker_directory}/launches/${launch_id}"
    mkdir -p "${launch_directory}"
    jobs_file="${launch_directory}/jobs.tsv"
fi

prepare_command=(
    "${julia_binary}" --startup-file=no "--project=${project_root}"
    "${script_directory}/prepare_corpus_workers.jl"
    --study "${study}" --split "${split}" --worker-dir "${worker_directory}"
    --expert-path "${expert_path}" --jobs-file "${jobs_file}"
)
[[ "${overwrite}" == false ]] || prepare_command+=(--overwrite)
"${prepare_command[@]}"

mapfile -t jobs < <(tail -n +2 "${jobs_file}")
if ((${#jobs[@]} == 0)); then
    echo "No missing ${study} distillation-corpus workers."
    exit 0
fi

worker_command() {
    local row="$1" row_study row_split base_seed mirror output_path command
    IFS=$'\t' read -r row_study row_split base_seed mirror output_path <<<"${row}"
    local command_parts=(
        "${julia_binary}" --startup-file=no "--project=${project_root}"
        "${script_directory}/run_corpus_worker.jl"
        --study "${row_study}" --split "${row_split}" --base-seed "${base_seed}"
        --mirror "${mirror}" --worker-dir "${worker_directory}"
        --expert-path "${expert_path}" --run-seed "${run_seed}"
    )
    [[ "${overwrite}" == false ]] || command_parts+=(--overwrite)
    printf -v command '%q ' "${command_parts[@]}"
    printf '%s' "${command}"
}

session_name() {
    local row="$1" row_study row_split base_seed mirror output_path
    IFS=$'\t' read -r row_study row_split base_seed mirror output_path <<<"${row}"
    printf 'distill_%s_%s_%s_m%s' "${row_study}" "${row_split}" "${base_seed}" "${mirror:0:1}"
}

if [[ "${preview}" == true ]]; then
    echo
    echo "Preview only; no tmux session was started."
    for row in "${jobs[@]}"; do
        printf '%s: ' "$(session_name "${row}")"
        worker_command "${row}"
        echo
    done
    exit 0
fi

command -v tmux >/dev/null 2>&1 || { echo "tmux is required." >&2; exit 1; }
worker_count=${#jobs[@]}
((worker_count > max_workers)) && worker_count=${max_workers}
first_session=""
started_count=0

for ((slot=0; slot<worker_count; slot++)); do
    slot_script="${launch_directory}/slot_$((slot + 1)).sh"
    first_row="${jobs[slot]}"
    session="$(session_name "${first_row}")"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -uo pipefail'
        printf 'export OPENBLAS_NUM_THREADS=%q\n' "${openblas_threads}"
        printf 'export OMP_NUM_THREADS=%q\n' "${omp_threads}"
        echo 'slot_failed=0'
        for ((index=slot; index<${#jobs[@]}; index+=worker_count)); do
            row="${jobs[index]}"
            job_name="$(session_name "${row}")"
            log_path="${launch_directory}/${job_name}.log"
            command="$(worker_command "${row}")"
            printf 'current_session=$(tmux display-message -p -t "$TMUX_PANE" "#S")\n'
            printf 'if [[ "$current_session" != %q ]]; then tmux rename-session -t "$current_session" %q; fi\n' "${job_name}" "${job_name}"
            printf 'echo %q\n' "Starting ${job_name}"
            printf '%s 2>&1 | tee %q || slot_failed=1\n' "${command}" "${log_path}"
        done
        echo 'exit "$slot_failed"'
    } > "${slot_script}"
    chmod u+x "${slot_script}"
    if tmux has-session -t "=${session}" 2>/dev/null; then
        echo "Skipping active tmux session ${session}"
        continue
    fi
    tmux new-session -d -s "${session}" "bash '${slot_script}'"
    [[ -n "${first_session}" ]] || first_session="${session}"
    started_count=$((started_count + 1))
    echo "Started ${session}"
done

echo
echo "Started ${started_count} persistent tmux slots for ${#jobs[@]} ${study} jobs."
echo "Manifest and logs: ${launch_directory}"
[[ -z "${first_session}" ]] || echo "Attach with: tmux attach -t ${first_session}"

