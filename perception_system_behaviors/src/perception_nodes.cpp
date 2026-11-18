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
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "tf2/exceptions.h"
#include "tf2_ros/transform_listener.h"

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
        "mode", "single", "single (one detection), group (their midpoint), each (a frame "
        "<frame_id>_<id> per detection) or follow (single, recognized again by its appearance)"),
      BT::InputPort<std::string>("class", "", "The class of the detections (each: empty, all)"),
      BT::InputPort<std::string>("id", "", "single, follow: the tracker id of the detection"),
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
  } else if (mode == "follow") {
    request.mode = srv::StartTargetFrame::Request::MODE_FOLLOW;
  } else {
    RCLCPP_ERROR(
      node_->get_logger(), "%s: mode must be single, group, each or follow", name().c_str());
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
      BT::InputPort<std::string>("skill", "lip_activity, facial_expression, body, markers"),
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
    BT::InputPort<double>("max_distance", 0.0, "Only detections nearer than this, m (0: any)"),
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
  double max_distance = 0.0;
  getInput("class", class_name);
  getInput("policy", policy);
  getInput("max_age_ms", max_age_ms);
  getInput("max_distance", max_distance);
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
    if (max_distance > 0.0 && distance > max_distance) {
      continue;
    }
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

// ---------------------------------------------------------------------------- FrameTargetSeen

FrameTargetSeen::FrameTargetSeen(const std::string & name, const BT::NodeConfig & config)
: BT::ConditionNode(name, config), node_(node_from_blackboard(config, name)) {}

BT::PortsList FrameTargetSeen::providedPorts()
{
  return {
    BT::InputPort<std::string>("frame_request", "The target frame request (MODE_FOLLOW)"),
    BT::InputPort<std::string>(
      "frames_namespace", "/perception_system/frames", "The frames_namespace of the system"),
    BT::InputPort<int>("max_age_ms", 1000, "How old the latest message may be"),
    BT::OutputPort<std::string>("id", "The current tracker id of the object"),
  };
}

BT::NodeStatus FrameTargetSeen::tick()
{
  std::string frame_request, frames_namespace = "/perception_system/frames";
  if (!required(*this, node_, "frame_request", frame_request)) {
    return BT::NodeStatus::FAILURE;
  }
  getInput("frames_namespace", frames_namespace);
  ids_.subscribe(node_, frames_namespace + "/" + frame_request + "/tracked_ids");
  int max_age_ms = 1000;
  getInput("max_age_ms", max_age_ms);
  auto msg = ids_.fresh(max_age_ms);
  if (!msg || msg->detections.empty() || msg->detections[0].id.empty()) {
    return BT::NodeStatus::FAILURE;
  }
  setOutput("id", msg->detections[0].id);
  return BT::NodeStatus::SUCCESS;
}

// ---------------------------------------------------------------------------- GestureSeen

std::string gesture_matching(const std::string & query, const std::vector<std::string> & made)
{
  for (const auto & gesture : made) {
    if (query.empty() || gesture == query ||
      (query == "hand_raised" && (gesture == "hand_raised_left" ||
      gesture == "hand_raised_right" || gesture == "both_hands_raised")) ||
      (query == "pointing" && (gesture == "pointing_left" || gesture == "pointing_right")))
    {
      return gesture;
    }
  }
  return "";
}

namespace
{

BT::PortsList people_ports(BT::PortsList more)
{
  return merge(
    {
      BT::InputPort<std::string>("topic", "/people/body", "The people of the body detector"),
      BT::InputPort<int>("max_age_ms", 1000, "How old the latest message may be"),
      BT::OutputPort<std::string>("person_id", "Who: their tracker id"),
      BT::OutputPort<std::string>("person_frame", "Their TF frame (empty if not placed)"),
      BT::OutputPort<int>("people", "How many people are in view"),
    }, more);
}

}  // namespace

GestureSeen::GestureSeen(const std::string & name, const BT::NodeConfig & config)
: BT::ConditionNode(name, config), node_(node_from_blackboard(config, name)) {}

BT::PortsList GestureSeen::providedPorts()
{
  return people_ports(
    {
      BT::InputPort<std::string>(
        "gesture", "", "hand_raised, hand_raised_left, hand_raised_right, both_hands_raised, "
        "waving, pointing, pointing_left, pointing_right (empty: any)"),
      BT::OutputPort<std::string>("made", "The gesture made"),
    });
}

BT::NodeStatus GestureSeen::tick()
{
  std::string topic = "/people/body", query;
  int max_age_ms = 1000;
  getInput("topic", topic);
  getInput("gesture", query);
  getInput("max_age_ms", max_age_ms);
  people_.subscribe(node_, topic);
  auto msg = people_.fresh(max_age_ms);
  setOutput("people", msg ? static_cast<int>(msg->people.size()) : 0);
  if (!msg) {
    return BT::NodeStatus::FAILURE;
  }
  for (const auto & person : msg->people) {
    const auto made = gesture_matching(query, person.gestures);
    if (!made.empty()) {
      setOutput("person_id", person.id);
      setOutput("person_frame", person.frame_id);
      setOutput("made", made);
      return BT::NodeStatus::SUCCESS;
    }
  }
  return BT::NodeStatus::FAILURE;
}

// ---------------------------------------------------------------------------- PersonFallen

PersonFallen::PersonFallen(const std::string & name, const BT::NodeConfig & config)
: BT::ConditionNode(name, config), node_(node_from_blackboard(config, name)) {}

BT::PortsList PersonFallen::providedPorts()
{
  return people_ports({});
}

BT::NodeStatus PersonFallen::tick()
{
  std::string topic = "/people/body";
  int max_age_ms = 1000;
  getInput("topic", topic);
  getInput("max_age_ms", max_age_ms);
  people_.subscribe(node_, topic);
  auto msg = people_.fresh(max_age_ms);
  setOutput("people", msg ? static_cast<int>(msg->people.size()) : 0);
  if (!msg) {
    return BT::NodeStatus::FAILURE;
  }
  for (const auto & person : msg->people) {
    if (person.fallen) {
      setOutput("person_id", person.id);
      setOutput("person_frame", person.frame_id);
      return BT::NodeStatus::SUCCESS;
    }
  }
  return BT::NodeStatus::FAILURE;
}

// ---------------------------------------------------------------------------- TF

std::shared_ptr<tf2_ros::Buffer> tf_buffer_for(const rclcpp::Node::SharedPtr & node)
{
  struct Entry
  {
    std::weak_ptr<rclcpp::Node> node;
    std::shared_ptr<tf2_ros::Buffer> buffer;
    std::shared_ptr<tf2_ros::TransformListener> listener;
  };
  static std::mutex mutex;
  static std::map<const rclcpp::Node *, Entry> entries;
  std::lock_guard<std::mutex> lock(mutex);
  auto & entry = entries[node.get()];
  // A node destroyed and another one created at its address: a new buffer and listener
  if (!entry.buffer || entry.node.lock() != node) {
    entry.node = node;
    entry.buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    entry.listener = std::make_shared<tf2_ros::TransformListener>(*entry.buffer, node, false);
  }
  return entry.buffer;
}

// ---------------------------------------------------------------------------- FrameInFront

FrameInFront::FrameInFront(const std::string & name, const BT::NodeConfig & config)
: BT::ConditionNode(name, config), node_(node_from_blackboard(config, name)),
  buffer_(tf_buffer_for(node_)) {}

BT::PortsList FrameInFront::providedPorts()
{
  return {
    BT::InputPort<std::string>("frame", "The TF frame (a person, an object)"),
    BT::InputPort<std::string>("reference_frame", "base_link", "The robot (x forward)"),
    BT::InputPort<double>("max_angle_deg", 5.0, "In front within this angle"),
    BT::OutputPort<double>("bearing", "Its angle from the x axis, rad (positive: left)"),
    BT::OutputPort<int>("direction", "Where to turn: 1 left, -1 right, 0 in front"),
    BT::OutputPort<double>("distance", "Its distance on the floor, m"),
  };
}

BT::NodeStatus FrameInFront::tick()
{
  std::string frame, reference = "base_link";
  double max_angle_deg = 5.0;
  if (!required(*this, node_, "frame", frame)) {
    return BT::NodeStatus::FAILURE;
  }
  getInput("reference_frame", reference);
  getInput("max_angle_deg", max_angle_deg);
  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = buffer_->lookupTransform(reference, frame, tf2::TimePointZero);
  } catch (const tf2::TransformException &) {
    setOutput("direction", 0);
    return BT::NodeStatus::FAILURE;
  }
  const auto & t = transform.transform.translation;
  const double bearing = std::atan2(t.y, t.x);
  const bool in_front = std::abs(bearing) <= max_angle_deg * M_PI / 180.0;
  setOutput("bearing", bearing);
  setOutput("distance", std::hypot(t.x, t.y));
  setOutput("direction", in_front ? 0 : (bearing > 0.0 ? 1 : -1));
  return in_front ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

// ---------------------------------------------------------------------------- MeasureDisplacement

MeasureDisplacement::MeasureDisplacement(const std::string & name, const BT::NodeConfig & config)
: BT::StatefulActionNode(name, config), node_(node_from_blackboard(config, name)),
  buffer_(tf_buffer_for(node_)) {}

BT::PortsList MeasureDisplacement::providedPorts()
{
  return {
    BT::InputPort<std::string>("frame", "The TF frame that moves (a followed person)"),
    BT::InputPort<std::string>("fixed_frame", "odom", "The floor (x, y)"),
    BT::InputPort<double>("start_distance", 0.2, "How far it moves to have started, m"),
    BT::InputPort<double>("distance", 0.0, "The distance to measure, m (0: no limit)"),
    BT::InputPort<unsigned>("duration_ms", 0, "Or the time to measure (0: no limit)"),
    BT::InputPort<unsigned>("lost_timeout_ms", 3000, "FAILURE if not updated for this long"),
    BT::OutputPort<std::string>("state", "WAITING_START or WALKING"),
    BT::OutputPort<double>("walked", "m, the straight line from the start"),
    BT::OutputPort<double>("elapsed", "s since the start"),
    BT::OutputPort<double>("speed", "m/s"),
    BT::OutputPort<double>("start_time", "s, the time of the image of the start"),
    BT::OutputPort<double>("end_time", "s, the time of the last image"),
  };
}

BT::NodeStatus MeasureDisplacement::onStart()
{
  has_origin_ = walking_ = false;
  last_update_ = node_->now();
  last_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  setOutput("state", std::string("WAITING_START"));
  setOutput("walked", 0.0);
  setOutput("elapsed", 0.0);
  setOutput("speed", 0.0);
  return onRunning();
}

BT::NodeStatus MeasureDisplacement::onRunning()
{
  std::string frame, fixed = "odom";
  double start_distance = 0.2, distance = 0.0;
  unsigned duration_ms = 0, lost_timeout_ms = 3000;
  if (!required(*this, node_, "frame", frame)) {
    return BT::NodeStatus::FAILURE;
  }
  getInput("fixed_frame", fixed);
  getInput("start_distance", start_distance);
  getInput("distance", distance);
  getInput("duration_ms", duration_ms);
  getInput("lost_timeout_ms", lost_timeout_ms);

  geometry_msgs::msg::TransformStamped transform;
  bool fresh = false;
  try {
    transform = buffer_->lookupTransform(fixed, frame, tf2::TimePointZero);
    const rclcpp::Time stamp(transform.header.stamp, RCL_ROS_TIME);
    fresh = stamp > last_stamp_;
    if (fresh) {
      last_stamp_ = stamp;
      last_update_ = node_->now();
    }
  } catch (const tf2::TransformException &) {
  }
  if ((node_->now() - last_update_).seconds() * 1000.0 > lost_timeout_ms) {
    RCLCPP_WARN(node_->get_logger(), "%s: %s not updated for %u ms", name().c_str(),
      frame.c_str(), lost_timeout_ms);
    return BT::NodeStatus::FAILURE;
  }
  if (!fresh) {
    return BT::NodeStatus::RUNNING;
  }
  const double x = transform.transform.translation.x, y = transform.transform.translation.y;
  if (!has_origin_) {
    has_origin_ = true;
    origin_x_ = x;
    origin_y_ = y;
    return BT::NodeStatus::RUNNING;
  }
  if (!walking_) {
    if (std::hypot(x - origin_x_, y - origin_y_) < start_distance) {
      return BT::NodeStatus::RUNNING;
    }
    walking_ = true;
    start_x_ = x;
    start_y_ = y;
    start_stamp_ = last_stamp_;
    setOutput("state", std::string("WALKING"));
    setOutput("start_time", start_stamp_.seconds());
  }
  const double walked = std::hypot(x - start_x_, y - start_y_);
  const double elapsed = (last_stamp_ - start_stamp_).seconds();
  setOutput("walked", walked);
  setOutput("elapsed", elapsed);
  setOutput("speed", elapsed > 0.0 ? walked / elapsed : 0.0);
  setOutput("end_time", last_stamp_.seconds());
  if ((distance > 0.0 && walked >= distance) ||
    (duration_ms > 0 && elapsed * 1000.0 >= duration_ms))
  {
    return BT::NodeStatus::SUCCESS;
  }
  return BT::NodeStatus::RUNNING;
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
  factory.registerNodeType<FrameTargetSeen>("FrameTargetSeen");
  factory.registerNodeType<GestureSeen>("GestureSeen");
  factory.registerNodeType<PersonFallen>("PersonFallen");
  factory.registerNodeType<FrameInFront>("FrameInFront");
  factory.registerNodeType<MeasureDisplacement>("MeasureDisplacement");
}
