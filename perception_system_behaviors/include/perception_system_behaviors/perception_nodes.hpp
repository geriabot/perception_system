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

#ifndef PERCEPTION_SYSTEM_BEHAVIORS__PERCEPTION_NODES_HPP_
#define PERCEPTION_SYSTEM_BEHAVIORS__PERCEPTION_NODES_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "behaviortree_cpp/action_node.h"
#include "behaviortree_cpp/condition_node.h"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

#include "perception_system_behaviors/AsyncServiceCall.hpp"
#include "perception_system_interfaces/msg/person_array.hpp"
#include "perception_system_interfaces/srv/detect_once.hpp"
#include "perception_system_interfaces/srv/start_detection.hpp"
#include "perception_system_interfaces/srv/start_skill.hpp"
#include "perception_system_interfaces/srv/start_target_frame.hpp"
#include "perception_system_interfaces/srv/stop_request.hpp"

namespace perception_system_behaviors
{

/// The rclcpp node of the tree, from the blackboard entry "node" (as the BehaviorRunner of
/// koch2 and the attention system set it), or "@node" (the root blackboard, from
/// a subtree).
rclcpp::Node::SharedPtr node_from_blackboard(
  const BT::NodeConfig & config, const std::string & who);

/**
 * A request to the perception system as a Behavior Tree action: the request is sent when the
 * service is available and the response checked on every tick, within timeout_ms. SUCCESS when
 * the perception system accepts it, FAILURE when it refuses it (the reason in the log) or does
 * not answer.
 */
template<typename ServiceT>
class PerceptionServiceAction : public BT::StatefulActionNode
{
public:
  PerceptionServiceAction(
    const std::string & name, const BT::NodeConfig & config,
    const std::string & default_service, std::chrono::milliseconds default_timeout)
  : BT::StatefulActionNode(name, config),
    node_(node_from_blackboard(config, name)),
    default_service_(default_service), default_timeout_(default_timeout) {}

  static BT::PortsList common_ports(const std::string & default_service, int timeout_ms)
  {
    return {
      BT::InputPort<std::string>("service_name", default_service, "The service to call"),
      BT::InputPort<int>("timeout_ms", timeout_ms, "Maximum wait for the service and its reply"),
    };
  }

protected:
  /// Fill the request from the ports; false (with a log) if a required port is missing.
  virtual bool fill(typename ServiceT::Request & request) = 0;
  /// The tree status for an accepted request (and the outputs to set).
  virtual BT::NodeStatus on_success(const typename ServiceT::Response & response)
  {
    (void)response;
    return BT::NodeStatus::SUCCESS;
  }

  BT::NodeStatus onStart() override
  {
    auto request = std::make_shared<typename ServiceT::Request>();
    if (!fill(*request)) {
      return BT::NodeStatus::FAILURE;
    }
    std::string service = default_service_;
    getInput("service_name", service);
    int timeout_ms = static_cast<int>(default_timeout_.count());
    getInput("timeout_ms", timeout_ms);
    call_.configure(node_, service);
    call_.start(request, std::chrono::milliseconds(timeout_ms));
    return BT::NodeStatus::RUNNING;
  }

  BT::NodeStatus onRunning() override
  {
    switch (call_.poll()) {
      case AsyncServiceCall<ServiceT>::Status::SUCCEEDED: {
          auto response = call_.response();
          if (!response->success) {
            RCLCPP_WARN(
              node_->get_logger(), "%s: the perception system refused the request: %s",
              name().c_str(), response->message.c_str());
            return BT::NodeStatus::FAILURE;
          }
          return on_success(*response);
        }
      case AsyncServiceCall<ServiceT>::Status::FAILED:
        return BT::NodeStatus::FAILURE;
      default:
        return BT::NodeStatus::RUNNING;
    }
  }

  void onHalted() override
  {
    call_.cancel();
  }

  rclcpp::Node::SharedPtr node_;
  std::string default_service_;
  std::chrono::milliseconds default_timeout_;
  AsyncServiceCall<ServiceT> call_;
};

/// Opens or updates a detection request (StartDetection) and outputs its topics.
class StartDetection
  : public PerceptionServiceAction<perception_system_interfaces::srv::StartDetection>
{
public:
  StartDetection(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();

protected:
  bool fill(perception_system_interfaces::srv::StartDetection::Request & request) override;
  BT::NodeStatus on_success(
    const perception_system_interfaces::srv::StartDetection::Response & response) override;
};

/// Publishes a TF frame at a tracked detection or at a group (StartTargetFrame).
class StartTargetFrame
  : public PerceptionServiceAction<perception_system_interfaces::srv::StartTargetFrame>
{
public:
  StartTargetFrame(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();

protected:
  bool fill(perception_system_interfaces::srv::StartTargetFrame::Request & request) override;
};

/// Keeps a skill of the perception system active (StartSkill).
class StartPerceptionSkill
  : public PerceptionServiceAction<perception_system_interfaces::srv::StartSkill>
{
public:
  StartPerceptionSkill(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();

protected:
  bool fill(perception_system_interfaces::srv::StartSkill::Request & request) override;
};

/// Ends a request of the perception system by its name (StopRequest).
class StopPerception
  : public PerceptionServiceAction<perception_system_interfaces::srv::StopRequest>
{
public:
  StopPerception(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();

protected:
  bool fill(perception_system_interfaces::srv::StopRequest::Request & request) override;
};

/// The detections of a class in the next image (DetectOnce): SUCCESS if at least min_count.
class DetectOnce
  : public PerceptionServiceAction<perception_system_interfaces::srv::DetectOnce>
{
public:
  DetectOnce(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();

protected:
  bool fill(perception_system_interfaces::srv::DetectOnce::Request & request) override;
  BT::NodeStatus on_success(
    const perception_system_interfaces::srv::DetectOnce::Response & response) override;
};

/// The latest message of a topic, received on the tree's node, and its age.
template<typename MessageT>
class LatestMessage
{
public:
  void subscribe(const rclcpp::Node::SharedPtr & node, const std::string & topic)
  {
    if (topic == topic_ && subscription_) {
      return;
    }
    node_ = node;
    topic_ = topic;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      last_.reset();
    }
    subscription_ = node->create_subscription<MessageT>(
      topic, rclcpp::QoS(10).reliable(), [this](typename MessageT::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_ = msg;
        received_ = node_->now();
      });
  }

  /// The latest message if it is at most max_age_ms old, else nullptr.
  typename MessageT::SharedPtr fresh(int max_age_ms)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_ ||
      (node_->now() - received_) > rclcpp::Duration(std::chrono::milliseconds(max_age_ms)))
    {
      return nullptr;
    }
    return last_;
  }

private:
  rclcpp::Node::SharedPtr node_;
  std::string topic_;
  typename rclcpp::Subscription<MessageT>::SharedPtr subscription_;
  std::mutex mutex_;
  typename MessageT::SharedPtr last_;
  rclcpp::Time received_;
};

/**
 * Whether at least min_count detections (of a class, and with a tracker id if given) are in the
 * latest message of a detections_2d topic, received less than max_age_ms ago. The topic usually
 * comes from the output detections_2d_topic of StartDetection. Outputs how many there are.
 */
class DetectionSeen : public BT::ConditionNode
{
public:
  DetectionSeen(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  LatestMessage<vision_msgs::msg::Detection2DArray> detections_;
};

/**
 * Chooses one tracked detection of a class in the latest message of a detections_3d topic:
 * the nearest to the camera or the first one. Outputs its tracker id. FAILURE if none.
 */
class SelectDetection : public BT::ConditionNode
{
public:
  SelectDetection(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  LatestMessage<vision_msgs::msg::Detection3DArray> detections_;
};

/**
 * Whether the object of a target frame request in MODE_FOLLOW is seen: the latest message of
 * <frames namespace>/<frame_request>/tracked_ids, received less than max_age_ms ago, has its id
 * (the id changes when the object is recognized again). Outputs the current id.
 */
class FrameTargetSeen : public BT::ConditionNode
{
public:
  FrameTargetSeen(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  LatestMessage<vision_msgs::msg::Detection2DArray> ids_;
};

/// Whether a gesture asked for is one of those made: "" any, "hand_raised" and "pointing" either
/// side, or the exact name. Returns the gesture made, or "".
std::string gesture_matching(const std::string & query, const std::vector<std::string> & made);

/**
 * Whether someone in the latest message of the body detector (/people/body, the skill "body")
 * makes a gesture (gesture: hand_raised, hand_raised_left, hand_raised_right, both_hands_raised,
 * waving, pointing, pointing_left, pointing_right; empty: any). Outputs who (person_id, the TF
 * frame) and the gesture made; always outputs how many people are in view.
 */
class GestureSeen : public BT::ConditionNode
{
public:
  GestureSeen(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  LatestMessage<perception_system_interfaces::msg::PersonArray> people_;
};

/**
 * Whether someone in the latest message of the body detector (/people/body) is fallen on the
 * floor. Outputs who (person_id, the TF frame); always outputs how many people are in view.
 */
class PersonFallen : public BT::ConditionNode
{
public:
  PersonFallen(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  LatestMessage<perception_system_interfaces::msg::PersonArray> people_;
};

/// The TF buffer of a node (one per node, with its listener), shared by the nodes of its trees.
std::shared_ptr<tf2_ros::Buffer> tf_buffer_for(const rclcpp::Node::SharedPtr & node);

/**
 * Whether a TF frame (a person, an object) is in front of the robot: its bearing from the x axis
 * of reference_frame (base_link), within max_angle_deg. Outputs the bearing (rad, positive to
 * the left), the side to turn to (direction: 1 left, -1 right, 0 in front) and the distance.
 * FAILURE if it is not in front, or the frame does not exist (direction 0). As IsInFront of
 * bt_nodes, for a behavior that turns the robot until it faces someone.
 */
class FrameInFront : public BT::ConditionNode
{
public:
  FrameInFront(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus tick() override;

private:
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> buffer_;
};

/**
 * How far a TF frame (a followed person) moves on the floor (x, y of fixed_frame), with the times
 * of its transforms (those of the images). It waits until the frame moves start_distance from
 * where it first was (state WAITING_START), then measures from that point and that time (state
 * WALKING) until it is distance away, or duration_ms have passed. Outputs the distance (m, the
 * straight line from the start), the elapsed time (s), the speed (m/s) and the start and end
 * times (s). SUCCESS when the distance or the duration is reached; FAILURE if the frame is not
 * updated for lost_timeout_ms. Runs until halted when distance and duration are 0.
 */
class MeasureDisplacement : public BT::StatefulActionNode
{
public:
  MeasureDisplacement(const std::string & name, const BT::NodeConfig & config);
  static BT::PortsList providedPorts();
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override {}

private:
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<tf2_ros::Buffer> buffer_;
  bool has_origin_ {false}, walking_ {false};
  double origin_x_ {0.0}, origin_y_ {0.0}, start_x_ {0.0}, start_y_ {0.0};
  rclcpp::Time start_stamp_, last_stamp_;
  rclcpp::Time last_update_;          // the wall time of the last new transform
};

}  // namespace perception_system_behaviors

#endif  // PERCEPTION_SYSTEM_BEHAVIORS__PERCEPTION_NODES_HPP_
