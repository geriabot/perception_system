# Prepares a terminal for the native installation (docs/INSTALLATION.md): system ROS 2, the
# workspace overlay and, optionally, a Python virtual environment. Source this file:
#
#     source <workspace>/src/perception_system/activate.sh
#
# Edit the two variables below once for your computer.

# The ROS 2 distribution installed under /opt/ros/<distro> (a ROS_DISTRO already set is kept)
: "${ROS_DISTRO:=jazzy}"

# Absolute path of the virtual environment where requirements.txt was installed; "" when the
# packages were installed in the system Python (--break-system-packages)
PERCEPTION_VENV=""

if [ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
else
  echo "perception_system: /opt/ros/${ROS_DISTRO}/setup.bash not found:" \
    "set ROS_DISTRO in activate.sh" >&2
fi

# The workspace overlay: this file is in <workspace>/src/perception_system/
_perception_ws="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"
if [ -f "${_perception_ws}/install/setup.bash" ]; then
  source "${_perception_ws}/install/setup.bash"
fi

# Sourcing the overlay reorders PYTHONPATH: the environment's packages go back in front
if [ -n "${PERCEPTION_VENV}" ]; then
  [ -f "${PERCEPTION_VENV}/bin/activate" ] && source "${PERCEPTION_VENV}/bin/activate"
  for _sp in "${PERCEPTION_VENV}"/lib/python*/site-packages; do
    [ -d "${_sp}" ] && export PYTHONPATH="${_sp}:${PYTHONPATH:-}"
  done
fi
unset _perception_ws _sp

echo "perception_system env ready — ROS ${ROS_DISTRO}"
