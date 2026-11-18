#!/usr/bin/env bash
# Copyright 2026 Rodrigo Pérez-Rodríguez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The checks of the skills body and markers (evaluation/README.md): the perception system without a
# camera of its own, and check_skills.py, which is the camera (photographs of people, a marker).
# Needs Ultralytics and the pose weights (downloaded once by the body detector). Run it in a shell
# with the workspace sourced, on a ROS domain that nothing else uses:
#
#   ROS_DOMAIN_ID=<free> bash src/perception_system/evaluation/scripts/run_skills_check.sh \
#     [out_dir] [launch args]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../results/skills_$(date +%Y%m%d_%H%M%S)}"
shift || true
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"

pids=()
stop_all() {
  for pid in "${pids[@]}"; do
    kill -INT -- "-$pid" 2>/dev/null
  done
  sleep 3
  for pid in "${pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null
  done
}
trap stop_all EXIT

setsid ros2 launch perception_system perception_system.launch.py camera:=none \
  model:=color_blobs device:=cpu "$@" > "$OUT/system.log" 2>&1 &
pids+=($!)
python3 "$HERE/check_skills.py" --report "$OUT/report_skills.json" | tee "$OUT/check_skills.log"
status=${PIPESTATUS[0]}
echo "Logs and report: $OUT"
exit "$status"
