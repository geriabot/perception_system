cmake_minimum_required(VERSION 3.10)
project(perception_system_behaviors)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(behaviortree_cpp REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(perception_system_interfaces REQUIRED)

# One plugin with every node: load it with the name perception_system_bt_nodes
add_library(perception_system_bt_nodes SHARED src/perception_nodes.cpp)
target_include_directories(perception_system_bt_nodes PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_compile_definitions(perception_system_bt_nodes PRIVATE BT_PLUGIN_EXPORT)
target_link_libraries(perception_system_bt_nodes PUBLIC
  rclcpp::rclcpp
  behaviortree_cpp::behaviortree_cpp
  ${vision_msgs_TARGETS}
  ${perception_system_interfaces_TARGETS})

install(TARGETS perception_system_bt_nodes
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(DIRECTORY behavior_tree_xml DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_perception_nodes test/test_perception_nodes.cpp TIMEOUT 60)
  target_link_libraries(test_perception_nodes perception_system_bt_nodes)
  target_compile_definitions(test_perception_nodes PRIVATE
    PLUGIN_PATH="$<TARGET_FILE:perception_system_bt_nodes>"
    BEHAVIORS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/behavior_tree_xml")
endif()

ament_export_include_directories(include/${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp behaviortree_cpp vision_msgs perception_system_interfaces)
ament_package()
