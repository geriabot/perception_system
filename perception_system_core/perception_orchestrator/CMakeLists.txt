cmake_minimum_required(VERSION 3.10)
project(perception_orchestrator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(koch2 REQUIRED)
find_package(perception_system_interfaces REQUIRED)

# The orchestrator, loaded by mission_executor from orchestrator_libraries
add_library(${PROJECT_NAME} SHARED src/perception_orchestrator.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
  ${koch2_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
  ${koch2_LIBRARIES}
  rclcpp::rclcpp
  rclcpp_action::rclcpp_action
  rclcpp_lifecycle::rclcpp_lifecycle
  behaviortree_cpp::behaviortree_cpp
  ${lifecycle_msgs_TARGETS}
  ${perception_system_interfaces_TARGETS})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
