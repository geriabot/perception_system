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

# The checks of the running system (evaluation/README.md): the synthetic camera, the perception
# system with the color_blobs model on the CPU, check_tasks.py and check_system.py. Run it in a
# shell with the workspace sourced (pixi shell, or the native installation), on a ROS domain that
# nothing else uses:
#
#   ROS_DOMAIN_ID=<free> bash src/perception_system/evaluation/scripts/run_check.sh \
#     [out_dir] [launch args]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../results/check_$(date +%Y%m%d_%H%M%S)}"
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

setsid python3 "$HERE/synthetic_camera.py" --rate 5 > "$OUT/camera.log" 2>&1 &
pids+=($!)
USER_NODES=true
[ "${CHECK_SKILLS:-1}" = "1" ] || USER_NODES=false
setsid ros2 launch perception_system perception_system.launch.py camera:=rgbd \
  model:=color_blobs device:=cpu user:=$USER_NODES "$@" > "$OUT/system.log" 2>&1 &
pids+=($!)

# The tasks of the orchestrator first (check_system.py ends by deactivating /perception_system).
# CHECK_TASKS=0: without them (orchestrator:=false). CHECK_SKILLS=0: without the skills about the
# user (no MediaPipe in the environment)
status=0
if [ "${CHECK_TASKS:-1}" = "1" ]; then
  python3 "$HERE/check_tasks.py" --report "$OUT/report_tasks.json" | tee "$OUT/check_tasks.log"
  status=${PIPESTATUS[0]}
fi
SKILLS=()
[ "${CHECK_SKILLS:-1}" = "1" ] && SKILLS=(--skills)
python3 "$HERE/check_system.py" "${SKILLS[@]}" --report "$OUT/report.json" | tee "$OUT/check.log"
[ "${PIPESTATUS[0]}" = "0" ] || status=1
echo "Logs and report: $OUT"
exit "$status"
