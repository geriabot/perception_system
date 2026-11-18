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

# The perception system against attention_perception on the same synthetic camera
# (evaluation/README.md, "Equivalence with the attention system"). The workspace sourced, and
# LEGACY_INSTALL the install folder of attention_perception (built from the attention system):
#
#   ROS_DOMAIN_ID=<free> LEGACY_INSTALL=<path>/install bash run_equivalence.sh [out_dir] [args]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-$HERE/../results/equivalence_$(date +%Y%m%d_%H%M%S)}"
shift || true
mkdir -p "$OUT"

# A run is inconclusive (exit 2) when the two systems did not receive the same images: it is
# repeated, up to ATTEMPTS times (default 3), each in OUT/attempt_<n>
if [ -z "${ATTEMPT:-}" ]; then
  for n in $(seq 1 "${ATTEMPTS:-3}"); do
    ATTEMPT=$n bash "${BASH_SOURCE[0]}" "$OUT/attempt_$n" "$@"
    status=$?
    [ "$status" = "2" ] || exit "$status"
  done
  exit 2
fi
: "${LEGACY_INSTALL:?set LEGACY_INSTALL to the install folder of attention_perception}"
set +u; source "$LEGACY_INSTALL/setup.bash"; set -u

pids=()
stop_all() {
  for pid in "${pids[@]}"; do kill -INT -- "-$pid" 2>/dev/null; done
  sleep 3
  for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null; done
}
trap stop_all EXIT

# The perception of the attention system, as attention_perception.launch.py starts it with
# projector_mode:=rgbd, its tracker parameters, and the color_blobs model on the CPU
TRACKER="-p tracking:=true -p tracker_type:=strongsort -p max_age:=45 -p min_hits:=3 \
  -p min_conf:=0.25 -p max_cos_dist:=0.25 -p max_iou_dist:=0.875 -p nn_budget:=80 \
  -p mc_lambda:=0.98 -p ema_alpha:=0.9 -p reid_weights:=osnet_x0_25_msmt17.pt"
setsid python3 "$HERE/legacy_detector.py" --ros-args -r __node:=detector \
  -p image_topic:=/image_rgb -p model:=color_blobs -p device:=cpu $TRACKER \
  > "$OUT/legacy_detector.log" 2>&1 &
pids+=($!)
setsid ros2 run detection_3d_estimators detection_3d_estimator --ros-args \
  -p estimator:=depth_image -p camera_info_topic:=/rgbd_camera/camera_info \
  -p depth_image_topic:=/rgbd_camera/depth_image -p camera_frame_is_optical:=true \
  > "$OUT/legacy_3d.log" 2>&1 &
pids+=($!)
# Its CMakeLists.txt gives no RUNTIME DESTINATION: without --symlink-install the TF publishers
# are installed in bin/ instead of lib/attention_perception_nodes/
legacy_executable() {
  local dir="$LEGACY_INSTALL/attention_perception_nodes"
  if [ -x "$dir/lib/attention_perception_nodes/$1" ]; then
    echo "$dir/lib/attention_perception_nodes/$1"
  else
    echo "$dir/bin/$1"
  fi
}
setsid "$(legacy_executable pub_single_detection_tf_node)" > "$OUT/legacy_single.log" 2>&1 &
pids+=($!)
setsid "$(legacy_executable pub_n_detections_same_class_tf_node)" > "$OUT/legacy_group.log" 2>&1 &
pids+=($!)

# The perception system, with the same model and tracker
setsid ros2 launch perception_system perception_system.launch.py camera:=rgbd model:=color_blobs \
  device:=cpu tracker:=strongsort > "$OUT/system.log" 2>&1 &
pids+=($!)

python3 "$HERE/compare_with_attention.py" --report "$OUT/equivalence.json" "$@" \
  | tee "$OUT/compare.log"
status=${PIPESTATUS[0]}
echo "Logs and report: $OUT"
# 2: inconclusive (the systems did not receive the same images); run_equivalence.sh is run
# again by the loop below, up to ATTEMPTS times
exit "$status"
