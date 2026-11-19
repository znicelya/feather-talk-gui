#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo 'Usage: bash tools/acceptance/run.sh PACKAGES_DIR VIDEO AUDIO cpu|cuda|vulkan|metal DEVICE_SUBSTRING NEW_OUTPUT_DIR' >&2
  exit 2
fi

absolute_file() { (cd -- "$(dirname -- "$1")" && printf '%s/%s\n' "$PWD" "$(basename -- "$1")"); }
export FT_SMOKE_PACKAGES="$(cd -- "$1" && pwd -P)"
export FT_SMOKE_VIDEO="$(absolute_file "$2")"
export FT_SMOKE_AUDIO="$(absolute_file "$3")"
export FT_SMOKE_BACKEND="$4" FT_SMOKE_DEVICE="$5"
mkdir -- "$6"
export FT_SMOKE_ROOT="$(cd -- "$6" && pwd -P)"

{
  uname -srm
  case "$(uname -s)" in
    Darwin) sw_vers; system_profiler SPHardwareDataType SPDisplaysDataType | grep -E 'Chip|Processor Name|Metal|Chipset Model' || true ;;
    Linux)
      if [[ -r /etc/os-release ]]; then cat /etc/os-release; fi
      if command -v lspci >/dev/null; then lspci -nnk | grep -A4 -Ei 'VGA|3D controller|Display controller' || true; fi
      if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true; fi
      if command -v vulkaninfo >/dev/null; then vulkaninfo --summary || true; fi ;;
  esac
} > "$FT_SMOKE_ROOT/system.log" 2>&1
cd -- "$(dirname -- "$0")/../.."
cargo test --offline --locked -p feather-talk-gui --test hardware_acceptance train_cancel_reopen_resume_export -- --ignored --nocapture 2>&1 | tee "$FT_SMOKE_ROOT/smoke.log"
