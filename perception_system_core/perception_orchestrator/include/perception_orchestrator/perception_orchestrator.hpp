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

#ifndef PERCEPTION_ORCHESTRATOR__PERCEPTION_ORCHESTRATOR_HPP_
#define PERCEPTION_ORCHESTRATOR__PERCEPTION_ORCHESTRATOR_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "koch2/base_orchestrator.hpp"
#include "lifecycle_msgs/srv/change_state.hpp"
#include "lifecycle_msgs/srv/get_state.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "perception_system_interfaces/action/follow_person.hpp"
#include "perception_system_interfaces/action/measure_walk.hpp"
#include "perception_system_interfaces/action/observe_class.hpp"
#include "perception_system_interfaces/action/track_object.hpp"
#include "perception_system_interfaces/action/wait_for_gesture.hpp"
#include "perception_system_interfaces/action/wait_for_object.hpp"
#include "perception_system_interfaces/action/watch_for_fall.hpp"
#include "perception_system_interfaces/srv/get_objects.hpp"
#include "perception_system_interfaces/srv/get_people.hpp"
#include "perception_system_interfaces/srv/stop_request.hpp"

namespace perception_orchestrator
{

/**
 * The coordination layer of the perception system: the tasks of the robot's behaviors (track an
 * object, follow a person, wait for an object, observe a class, wait for a gesture, watch for a
 * fall) as ROS 2 actions, each run by a perception
 * behavior (a Behavior Tree in a BehaviorRunner) on the requests of /perception_system.
 *
 * Several clients may run tasks at the same time: each behavior has a pool of runners (slots,
 * parameter slots_per_behavior), one per goal in progress. A goal gets a free slot, the slot's
 * tree is built for that goal (its inputs written in the XML, its outputs in keys of the slot),
 * and the slot's runner is activated. The status of each runner is read from the runner itself
 * (get_bt_status()), not from /behavior_status, which every runner of the robot shares. When the
 * goal ends (its tree ends, it is cancelled, or the orchestrator is deactivated), the runner is
 * deactivated and the requests of the slot are ended on /perception_system.
 *
 * The orchestrator activates /perception_system in cascade (parameter manage_perception_system)
 * and accepts goals once it is active.
 */
class PerceptionOrchestrator : public koch2::BaseOrchestrator
{
public:
  explicit PerceptionOrchestrator(BT::Blackboard::Ptr blackboard);

  using TrackObject = perception_system_interfaces::action::TrackObject;
  using WaitForObject = perception_system_interfaces::action::WaitForObject;
  using ObserveClass = perception_system_interfaces::action::ObserveClass;
  using FollowPerson = perception_system_interfaces::action::FollowPerson;
  using WaitForGesture = perception_system_interfaces::action::WaitForGesture;
  using WatchForFall = perception_system_interfaces::action::WatchForFall;
  using MeasureWalk = perception_system_interfaces::action::MeasureWalk;

protected:
  enum class State : int {STARTING_PERCEPTION, READY};

  void control_cycle() override;
  void go_to_state(int state) override;

  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

private:
  /// A runner of a behavior and the goal it serves
  struct Slot
  {
    // The runner, e.g. perception_track_object_0 (runner names are unique in the robot); also
    // the name of its detection request
    std::string name;
    std::string behavior;    // track_object, wait_for_object, observe_class, follow_person, ...
    bool busy {false};
    bool ending {false};     // waiting for the last objects before the result
    // Set when the goal starts, for its action type
    std::function<bool()> canceling;
    std::function<void()> feedback;
    std::function<void(BT::NodeStatus)> finish;
    std::function<void(const std::string &)> abort;
    rclcpp::Time last_feedback;
    // The last GetObjects answer of the slot's detection request
    perception_system_interfaces::srv::GetObjects::Response::SharedPtr objects;
    bool objects_pending {false};
    // FollowPerson: the id of the last feedback and how many times it changed
    std::string last_id;
    int id_changes {0};
  };

  template<typename ActionT>
  void serve(const std::string & action_name, const std::string & behavior);

  rclcpp_action::GoalResponse accept(const std::string & behavior, const std::string & error);
  Slot * free_slot(const std::string & behavior);
  bool start(
    Slot & slot, const std::string & subtree, const std::map<std::string, std::string> & inputs,
    const std::map<std::string, unsigned> & durations, const std::vector<std::string> & outputs,
    const std::vector<std::string> & int_outputs = {},
    const std::map<std::string, double> & reals = {},
    const std::vector<std::string> & real_outputs = {});
  double real_output(const Slot & slot, const std::string & port) const;
  void end(Slot & slot);
  void request_objects(Slot & slot, bool only_tracked);
  void stop_requests(const Slot & slot);
  std::string key(const Slot & slot, const std::string & port) const;
  std::string output(const Slot & slot, const std::string & port) const;
  void manage_perception_system();

  void start_track_object(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      TrackObject>> handle);
  void start_wait_for_object(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      WaitForObject>> handle);
  void start_observe_class(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      ObserveClass>> handle);
  void start_follow_person(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      FollowPerson>> handle);
  void start_wait_for_gesture(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      WaitForGesture>> handle);
  void start_watch_for_fall(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      WatchForFall>> handle);
  void start_measure_walk(Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<
      MeasureWalk>> handle);
  /// The person with this id in the last image of the body detector, then done(found, person)
  void request_person(
    const std::string & id,
    std::function<void(bool, const perception_system_interfaces::msg::Person &)> done);

  State state_ {State::STARTING_PERCEPTION};
  bool active_ {false};
  std::map<std::string, Slot> slots_;
  std::vector<std::shared_ptr<void>> action_servers_;
  rclcpp::Client<perception_system_interfaces::srv::GetObjects>::SharedPtr get_objects_;
  rclcpp::Client<perception_system_interfaces::srv::StopRequest>::SharedPtr stop_;
  rclcpp::Client<perception_system_interfaces::srv::GetPeople>::SharedPtr get_people_;
  rclcpp::Client<lifecycle_msgs::srv::GetState>::SharedPtr perception_get_state_;
  rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedPtr perception_change_state_;
  bool perception_call_pending_ {false};
  rclcpp::Time last_perception_check_;
  std::string system_;       // the namespace of the services of /perception_system
  int pending_goals_ {0};
};

}  // namespace perception_orchestrator

#endif  // PERCEPTION_ORCHESTRATOR__PERCEPTION_ORCHESTRATOR_HPP_
