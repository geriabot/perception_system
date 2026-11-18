cmake_minimum_required(VERSION 3.8)
project(perception_system_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PerceptionStatus.msg"
  "msg/RequestStatus.msg"
  "msg/TrackedObject.msg"
  "msg/UserState.msg"
  "srv/DetectOnce.srv"
  "srv/GetObjects.srv"
  "srv/GetStatus.srv"
  "srv/GetUserState.srv"
  "srv/ListModels.srv"
  "srv/StartDetection.srv"
  "srv/StartSkill.srv"
  "srv/StartTargetFrame.srv"
  "srv/StopRequest.srv"
  "action/ObserveClass.action"
  "action/TrackObject.action"
  "action/WaitForObject.action"
  DEPENDENCIES builtin_interfaces geometry_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
