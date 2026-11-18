// Copyright 2026 Rodrigo Pérez-Rodríguez
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "perception_system_behaviors/perception_nodes.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"

namespace perception_system_behaviors
{

using namespace std::chrono_literals;
namespace srv = perception_system_interfaces::srv;

rclcpp::Node::SharedPtr
node_from_blackboard(const BT::NodeConfig & config, const std::string & who)
{
  rclcpp::Node::SharedPtr node;
  if (!config.blackboard ||
    !((config.blackboard->get("node", node) && node) ||
    (config.blackboard->get("@node", node) && node)))
  {
    throw BT::RuntimeError(who + ": 'node' (rclcpp::Node::SharedPtr) not found in the blackboard");
  }
  return node;
}

namespace
{

/// "a;b; c" -> {"a", "b", "c"}
std::vector<std::string> split(const std::string & text)
{
  std::vector<std::string> out;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ';')) {
    const auto first = item.find_first_not_of(' ');
    const auto last = item.find_last_not_of(' ');
    if (first != std::string::npos) {
      out.push_back(item.substr(first, last - first + 1));
    }
  }
  return out;
}

BT::PortsList merge(BT::PortsList ports, const BT::PortsList & more)
{
  ports.insert(more.begin(), more.end());
  return ports;
}

bool required(BT::TreeNode & node, const rclcpp::Node::SharedPtr & ros, const std::string & port,
  std::string & value)
{
  auto result = node.getInput<std::string>(port);
  if (!result || result.value().empty()) {
    RCLCPP_ERROR(
      ros->get_logger(), "%s: the input port '%s' is missing or empty", node.name().c_str(),
      port.c_str());
    return false;
  }
  value = result.value();
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------- StartDetection

StartDetection::StartDetection(const std::string & name, const BT::NodeConfig & config)
: PerceptionServiceAction(name, config, "/perception_system/start_detection", 30000ms) {}

BT::PortsList StartDetection::providedPorts()
{
  return merge(
    common_ports("/perception_system/start_detection", 30000), {
      BT::InputPort<std::string>("request", "The name of the request (unique per client)"),
      BT::InputPort<std::string>("class", "", "The class to detect (empty: images only)"),
      BT::InputPort<std::string>("category", "", "Its base category, also searched"),
      BT::InputPort<std::string>("labels", "", "Several classes instead, \"cup; bowl\""),
      BT::InputPort<std::string>(
        "output_namespace", "", "Where its topics go (empty: /perception_system/requests/"
        "<request>; \"/\": the root)"),
      BT::InputPort<int>("outputs", 0, "Sum of 1 (2D), 2 (3D), 4 (image), 8 (debug); 0: 1+2+4"),
      BT::InputPort<std::string>("model", "", "Detection model (empty: the default)"),
      BT::InputPort<std::string>("model_options", "", "Its options, YAML"),
      BT::InputPort<std::string>("tracker", "", "strongsort, bytetrack or none (empty: default)"),
      BT::OutputPort<std::string>("detections_2d_topic"),
      BT::OutputPort<std::string>("detections_3d_topic"),
      BT::OutputPort<std::string>("detections_image_topic"),
    });
}

bool StartDetection::fill(srv::StartDetection::Request & request)
{
  if (!required(*this, node_, "request", request.name)) {
    return false;
  }
  getInput("class", request.class_name);
  getInput("category", request.category);
  std::string labels;
  getInput("labels", labels);
  request.labels = split(labels);
  getInput("output_namespace", request.output_namespace);
  int outputs = 0;
  getInput("outputs", outputs);
  request.outputs = static_cast<uint8_t>(outputs);
  getInput("model", request.model);
  getInput("model_options", request.model_options);
  getInput("tracker", request.tracker);
  return true;
}

BT::NodeStatus StartDetection::on_success(const srv::StartDetection::Response & response)
{
  setOutput("detections_2d_topic", response.detections_2d_topic);
  setOutput("detections_3d_topic", response.detections_3d_topic);
  setOutput("detections_image_topic", response.detections_image_topic);
  return BT::NodeStatus::SUCCESS;
}

// ---------------------------------------------------------------------------- StartTargetFrame

StartTargetFrame::StartTargetFrame(const std::string & name, const BT::NodeConfig & config)
: PerceptionServiceAction(name, config, "/perception_system/start_target_frame", 5000ms) {}

BT::PortsList StartTargetFrame::providedPorts()
{
  return merge(
    common_ports("/perception_system/start_target_frame", 5000), {
      BT::InputPort<std::string>("request", "The name of the request"),
      BT::InputPort<std::string>("detection_request", "The detection request it follows"),
      BT::InputPort<std::string>("frame_id", "The TF frame to publish"),
      BT::InputPort<std::string>("fixed_frame_id", "", "Its parent (empty: odom)"),
      BT::InputPort<std::string>(
        "mode", "single", "single (one detection), group (their midpoint) or each (a frame "
        "<frame_id>_<id> per detection)"),
      BT::InputPort<std::string>("class", "", "The class of the detections (each: empty, all)"),
      BT::InputPort<std::string>("id", "", "single: the tracker id of the detection"),
      BT::InputPort<int>("n_detections", 0, "group: how many (0: all of the class)"),
      BT::InputPort<std::string>("half_width_topic", "", "group: topic of its half width"),
    });
}

bool StartTargetFrame::fill(srv::StartTargetFrame::Request & request)
{
  if (!required(*this, node_, "request", request.name) ||
    !required(*this, node_, "detection_request", request.detection_request) ||
    !required(*this, node_, "frame_id", request.frame_id))
  {
    return false;
  }
  getInput("class", request.class_name);
  getInput("fixed_frame_id", request.fixed_frame_id);
  std::string mode = "single";
  getInput("mode", mode);
  if (mode == "single") {
    request.mode = srv::StartTargetFrame::Request::MODE_SINGLE;
  } else if (mode == "group") {
    request.mode = srv::StartTargetFrame::Request::MODE_GROUP;
  } else if (mode == "each") {
    request.mode = srv::StartTargetFrame::Request::MODE_EACH;
  } else {
    RCLCPP_ERROR(node_->get_logger(), "%s: mode must be single, group or each", name().c_str());
    return false;
  }
  if (request.mode != srv::StartTargetFrame::Request::MODE_EACH && request.class_name.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "%s: the input port 'class' is empty", name().c_str());
    return false;
  }
  getInput("id", request.id);
  getInput("n_detections", request.n_detections);
  getInput("half_width_topic", request.half_width_topic);
  return true;
}

// ---------------------------------------------------------------------------- StartPerceptionSkill

StartPerceptionSkill::StartPerceptionSkill(const std::string & name, const BT::NodeConfig & config)
: PerceptionServiceAction(name, config, "/perception_system/start_skill", 15000ms) {}

BT::PortsList StartPerceptionSkill::providedPorts()
{
  return merge(
    common_ports("/perception_system/start_skill", 15000), {
      BT::InputPort<std::string>("request", "The name of the request"),
      BT::InputPort<std::string>("skill", "speaking, emotion, ..."),
    });
}

bool StartPerceptionSkill::fill(srv::StartSkill::Request & request)
{
  return required(*this, node_, "request", request.name) &&
         required(*this, node_, "skill", request.skill);
}

// ---------------------------------------------------------------------------- StopPerception

StopPerception::StopPerception(const std::string & name, const BT::NodeConfig & config)
: PerceptionServiceAction(name, config, "/perception_system/stop", 5000ms) {}

BT::PortsList StopPerception::providedPorts()
{
  return merge(
    common_ports("/perception_system/stop", 5000), {
      BT::InputPort<std::string>("request", "The name of the request to end"),
    });
}

bool StopPerception::fill(srv::StopRequest::Request & request)
{
  return required(*this, node_, "request", request.name);
}

// ---------------------------------------------------------------------------- DetectOnce

DetectOnce::DetectOnce(const std::string & name, const BT::NodeConfig & config)
: PerceptionServiceAction(name, config, "/perception_system/detect_once", 10000ms) {}

BT::PortsList DetectOnce::providedPorts()
{
  return merge(
    common_ports("/perception_system/detect_once", 10000), {
      BT::InputPort<std::string>("class", "", "The class to detect"),
      BT::InputPort<std::string>("category", "", "Its base category, also searched"),
      BT::InputPort<std::string>("labels", "", "Several classes instead, \"cup; bowl\""),
      BT::InputPort<std::string>("model", "", "Detection model (empty: the default)"),
      BT::InputPort<int>("min_count", 1, "SUCCESS if at least this many detections"),
      BT::OutputPort<int>("count", "How many detections"),
    });
}

bool DetectOnce::fill(srv::DetectOnce::Request & request)
{
  getInput("class", request.class_name);
  getInput("category", request.category);
  std::string labels;
  getInput("labels", labels);
  request.labels = split(labels);
  getInput("model", request.model);
  if (request.class_name.empty() && request.labels.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "%s: no class or labels", name().c_str());
    return false;
  }
  int timeout_ms = 10000;
  getInput("timeout_ms", timeout_ms);
  // The perception system waits a little less than the tree, so that its answer arrives
  request.timeout_s = std::max(0.5f, static_cast<float>(timeout_ms) / 1000.0f - 1.0f);
  return true;
}

BT::NodeStatus DetectOnce::on_success(const srv::DetectOnce::Response & response)
{
  const int count = static_cast<int>(response.detections_2d.detections.size());
  setOutput("count", count);
  int min_count = 1;
  getInput("min_count", min_count);
  return count >= min_count ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

// ---------------------------------------------------------------------------- DetectionSeen

DetectionSeen::DetectionSeen(const std::string & name, const BT::NodeConfig & config)
: BT::ConditionNode(name, config), node_(node_from_blackboard(config, name)) {}

BT::PortsList DetectionSeen::providedPorts()
{
  return {
    BT::InputPort<std::string>("topic", "A detections_2d topic of a detection request"),
    BT::InputPort<std::string>("class", "", "The class (empty: any)"),
    BT::InputPort<std::string>("id", "", "The tracker id (empty: any)"),
    BT::InputPort<int>("min_count", 1, "How many detections at least"),
    BT::InputPort<int>("max_age_ms", 1000, "How old the latest message may be"),
    BT::OutputPort<int>("count", "How many detections match"),
  };
}

BT::NodeStatus DetectionSeen::tick()
{
  std::string topic;
  if (!required(*this, node_, "topic", topic)) {
    return BT::NodeStatus::FAILURE;
  }
  detections_.subscribe(node_, topic);
  std::string class_name, id;
  int min_count = 1, max_age_ms = 1000;
  getInput("class", class_name);
  getInput("id", id);
  getInput("min_count", min_count);
  getInput("max_age_ms", max_age_ms);
  int count = 0;
  if (auto msg = detections_.fresh(max_age_ms)) {
    for (const auto & detection : msg->detections) {
      if (!detection.results.empty() &&
        (class_name.empty() || detection.results[0].hypothesis.class_id == class_name) &&
        (id.empty() || detection.id == id))
      {
        ++count;
      }
    }
  }
  setOutput("count", count);
  return count >= std::max(1, min_count) ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

// ---------------------------------------------------------------------------- SelectDetection

SelectDetection::SelectDetection(const std::string & name, const BT::NodeConfig & config)
: BT::ConditionNode(name, config), node_(node_from_blackboard(config, name)) {}

BT::PortsList SelectDetection::providedPorts()
{
  return {
    BT::InputPort<std::string>("topic", "A detections_3d topic of a detection request"),
    BT::InputPort<std::string>("class", "", "The class (empty: any)"),
    BT::InputPort<std::string>("policy", "nearest", "nearest (to the camera) or first"),
    BT::InputPort<int>("max_age_ms", 1000, "How old the latest message may be"),
    BT::OutputPort<std::string>("id", "The tracker id of the detection chosen"),
  };
}

BT::NodeStatus SelectDetection::tick()
{
  std::string topic;
  if (!required(*this, node_, "topic", topic)) {
    return BT::NodeStatus::FAILURE;
  }
  detections_.subscribe(node_, topic);
  std::string class_name, policy = "nearest";
  int max_age_ms = 1000;
  getInput("class", class_name);
  getInput("policy", policy);
  getInput("max_age_ms", max_age_ms);
  if (policy != "nearest" && policy != "first") {
    RCLCPP_ERROR(node_->get_logger(), "%s: policy must be nearest or first", name().c_str());
    return BT::NodeStatus::FAILURE;
  }
  auto msg = detections_.fresh(max_age_ms);
  if (!msg) {
    return BT::NodeStatus::FAILURE;
  }
  const vision_msgs::msg::Detection3D * chosen = nullptr;
  double best = 0.0;
  for (const auto & detection : msg->detections) {
    if (detection.id.empty() || detection.results.empty() ||
      (!class_name.empty() && detection.results[0].hypothesis.class_id != class_name))
    {
      continue;
    }
    const auto & p = detection.bbox.center.position;
    const double distance = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (!chosen || (policy == "nearest" && distance < best)) {
      chosen = &detection;
      best = distance;
    }
  }
  if (!chosen) {
    return BT::NodeStatus::FAILURE;
  }
  setOutput("id", chosen->id);
  return BT::NodeStatus::SUCCESS;
}

}  // namespace perception_system_behaviors

BT_REGISTER_NODES(factory)
{
  using namespace perception_system_behaviors;
  factory.registerNodeType<StartDetection>("StartDetection");
  factory.registerNodeType<StartTargetFrame>("StartTargetFrame");
  factory.registerNodeType<StartPerceptionSkill>("StartPerceptionSkill");
  factory.registerNodeType<StopPerception>("StopPerception");
  factory.registerNodeType<DetectOnce>("DetectOnce");
  factory.registerNodeType<DetectionSeen>("DetectionSeen");
  factory.registerNodeType<SelectDetection>("SelectDetection");
}
