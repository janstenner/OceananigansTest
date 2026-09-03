#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_directory}/launch_tmux_common.sh" ra5e4 "$@"
