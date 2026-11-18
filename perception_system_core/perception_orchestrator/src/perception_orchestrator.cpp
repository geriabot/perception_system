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

#include "perception_orchestrator/perception_orchestrator.hpp"

#include <cctype>
#include <sstream>
#include <tuple>
#include <type_traits>

#include "koch2/orchestrator_factory.hpp"
#include "lifecycle_msgs/msg/state.hpp"
#include "lifecycle_msgs/msg/transition.hpp"

namespace perception_orchestrator
{

using namespace std::chrono_literals;
using perception_system_interfaces::msg::TrackedObject;
using perception_system_interfaces::srv::GetObjects;
using perception_system_interfaces::srv::StopRequest;

namespace
{

// Timeout of a Behavior Tree node for "no limit" (about 23 days)
constexpr unsigned NO_LIMIT_MS = 2000000000;
const std::vector<std::string> BEHAVIORS = {"track_object", "wait_for_object", "observe_class"};

std::string escape(const std::string & text)
{
  std::string out;
  for (char c : text) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      default: out += c;
    }
  }
  return out;
}

/// "person in red" -> "person_in_red", for frame names
std::string frame_name(const std::string & text)
{
  std::string out;
  for (unsigned char c : text) {
    out += std::isalnum(c) ? static_cast<char>(std::tolower(c)) : '_';
  }
  return out;
}

unsigned milliseconds(float seconds, unsigned if_zero)
{
  return seconds > 0.0f ? static_cast<unsigned>(seconds * 1000.0f) : if_zero;
}

std::string goal_error(const PerceptionOrchestrator::TrackObject::Goal & goal)
{
  if (goal.class_name.empty()) {
    return "class_name is empty";
  }
  if (!goal.selection.empty() && goal.selection != "nearest" && goal.selection != "first") {
    return "selection must be nearest or first";
  }
  if (goal.find_timeout_s < 0.0f || goal.lost_timeout_s < 0.0f) {
    return "negative timeout";
  }
  return "";
}

std::string goal_error(const PerceptionOrchestrator::WaitForObject::Goal & goal)
{
  if (goal.class_name.empty()) {
    return "class_name is empty";
  }
  if (goal.min_count < 0 || goal.timeout_s < 0.0f) {
    return "negative min_count or timeout";
  }
  return "";
}

std::string goal_error(const PerceptionOrchestrator::ObserveClass::Goal & goal)
{
  if (goal.class_name.empty() && goal.labels.empty()) {
    return "no class_name and no labels";
  }
  return "";
}

}  // namespace

PerceptionOrchestrator::PerceptionOrchestrator(BT::Blackboard::Ptr blackboard)
: BaseOrchestrator("perception_orchestrator", blackboard)
{
  const int slots = declare_parameter<int>("slots_per_behavior", 3);
  const int tick_ms = declare_parameter<int>("tick_period_ms", 50);
  system_ = declare_parameter<std::string>("perception_system", "/perception_system");
  declare_parameter<bool>("manage_perception_system", true);
  declare_parameter<int>("feedback_period_ms", 500);
  const auto plugin = declare_parameter<std::string>(
    "plugin_library", "libperception_system_bt_nodes.so");

  // A pool of runners per behavior: their trees are given for each goal (start())
  for (const auto & behavior : BEHAVIORS) {
    for (int i = 0; i < slots; ++i) {
      const std::string name = "perception_" + behavior + "_" + std::to_string(i);
      auto runner = std::make_shared<koch2::BehaviorRunner>(
        blackboard_, name, "", std::vector<std::string>{plugin}, "perception_system_behaviors",
        tick_ms);
      register_runner(name, runner);
      Slot slot;
      slot.name = name;
      slot.behavior = behavior;
      slot.last_feedback = now();
      slots_[name] = slot;
    }
  }

  get_objects_ = create_client<GetObjects>(system_ + "/get_objects");
  stop_ = create_client<StopRequest>(system_ + "/stop");
  perception_get_state_ = create_client<lifecycle_msgs::srv::GetState>(system_ + "/get_state");
  perception_change_state_ =
    create_client<lifecycle_msgs::srv::ChangeState>(system_ + "/change_state");

  serve<TrackObject>(system_ + "/track_object", "track_object");
  serve<WaitForObject>(system_ + "/wait_for_object", "wait_for_object");
  serve<ObserveClass>(system_ + "/observe_class", "observe_class");
}

template<typename ActionT>
void PerceptionOrchestrator::serve(const std::string & action_name, const std::string & behavior)
{
  using GoalHandle = rclcpp_action::ServerGoalHandle<ActionT>;
  auto server = rclcpp_action::create_server<ActionT>(
    this, action_name,
    [this, behavior](const rclcpp_action::GoalUUID &,
    std::shared_ptr<const typename ActionT::Goal> goal) {
      return accept(behavior, goal_error(*goal));
    },
    [](const std::shared_ptr<GoalHandle>) {return rclcpp_action::CancelResponse::ACCEPT;},
    [this, behavior](const std::shared_ptr<GoalHandle> handle) {
      pending_goals_ = std::max(0, pending_goals_ - 1);
      Slot * slot = free_slot(behavior);
      if (slot == nullptr) {
        auto result = std::make_shared<typename ActionT::Result>();
        result->reason = "no free slot";
        handle->abort(result);
        return;
      }
      if constexpr (std::is_same_v<ActionT, TrackObject>) {
        start_track_object(*slot, handle);
      } else if constexpr (std::is_same_v<ActionT, WaitForObject>) {
        start_wait_for_object(*slot, handle);
      } else {
        start_observe_class(*slot, handle);
      }
    });
  action_servers_.push_back(server);
}

rclcpp_action::GoalResponse
PerceptionOrchestrator::accept(const std::string & behavior, const std::string & error)
{
  std::string reason = error;
  if (reason.empty() && (!active_ || state_ != State::READY)) {
    reason = "the perception system is not active yet";
  }
  int free = 0;
  for (const auto & [name, slot] : slots_) {
    free += slot.behavior == behavior && !slot.busy;
  }
  if (reason.empty() && free - pending_goals_ <= 0) {
    reason = "every " + behavior + " slot is busy (parameter slots_per_behavior)";
  }
  if (!reason.empty()) {
    RCLCPP_WARN(get_logger(), "%s goal rejected: %s", behavior.c_str(), reason.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }
  ++pending_goals_;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

PerceptionOrchestrator::Slot * PerceptionOrchestrator::free_slot(const std::string & behavior)
{
  for (auto & [name, slot] : slots_) {
    if (slot.behavior == behavior && !slot.busy) {
      return &slot;
    }
  }
  return nullptr;
}

std::string PerceptionOrchestrator::key(const Slot & slot, const std::string & port) const
{
  return slot.name + "_" + port;
}

std::string PerceptionOrchestrator::output(const Slot & slot, const std::string & port) const
{
  std::string value;
  if (!blackboard_->get(key(slot, port), value)) {
    return "";
  }
  return value;
}

bool PerceptionOrchestrator::start(
  Slot & slot, const std::string & subtree, const std::map<std::string, std::string> & inputs,
  const std::map<std::string, unsigned> & durations, const std::vector<std::string> & outputs,
  const std::vector<std::string> & int_outputs)
{
  // The slot's tree: the behavior as a subtree, its inputs written in the XML, its outputs in
  // keys of the slot (so that the goals of other slots, sharing the blackboard, do not mix). The
  // durations go in keys of the slot too, typed as the msec of Timeout (unsigned): a literal
  // number in the XML would be typed int by the subtree
  std::ostringstream xml;
  xml << "<root BTCPP_format=\"4\" main_tree_to_execute=\"Slot_" << slot.name << "\">"
      << "<include ros_pkg=\"perception_system_behaviors\" path=\"behavior_tree_xml/"
      << slot.behavior << ".xml\"/>"
      << "<BehaviorTree ID=\"Slot_" << slot.name << "\"><SubTree ID=\"" << subtree << "\"";
  for (const auto & [port, value] : inputs) {
    xml << " " << port << "=\"" << escape(value) << "\"";
  }
  for (const auto & [port, value] : durations) {
    blackboard_->set<unsigned>(key(slot, port), value);
    xml << " " << port << "=\"{" << key(slot, port) << "}\"";
  }
  // Each output key is created with the type its nodes write (a mismatch throws in the tree)
  for (const auto & port : outputs) {
    blackboard_->set(key(slot, port), std::string());
    xml << " " << port << "=\"{" << key(slot, port) << "}\"";
  }
  for (const auto & port : int_outputs) {
    blackboard_->set(key(slot, port), 0);
    xml << " " << port << "=\"{" << key(slot, port) << "}\"";
  }
  xml << "/></BehaviorTree></root>";
  auto runner = std::dynamic_pointer_cast<koch2::BehaviorRunner>(
    runners_.at(slot.name));
  runner->set_bt(xml.str());
  slot.busy = true;
  slot.ending = false;
  slot.objects.reset();
  slot.objects_pending = false;
  slot.last_feedback = now();
  if (!activate_runner(slot.name)) {
    slot.abort("the behavior could not be started: " + runner->activation_error());
    end(slot);
    return false;
  }
  RCLCPP_INFO(get_logger(), "%s: %s started", slot.name.c_str(), subtree.c_str());
  return true;
}

void PerceptionOrchestrator::end(Slot & slot)
{
  if (active_runners_.count(slot.name)) {
    deactivate_runner(slot.name);
  }
  stop_requests(slot);
  slot.busy = false;
  slot.ending = false;
  slot.canceling = nullptr;
  slot.feedback = nullptr;
  slot.finish = nullptr;
  slot.abort = nullptr;
  slot.objects.reset();
}

void PerceptionOrchestrator::stop_requests(const Slot & slot)
{
  // The detection request of the slot (its target frames end with it)
  if (!stop_->service_is_ready()) {
    RCLCPP_WARN(get_logger(), "%s/stop not available: %s may stay open", system_.c_str(),
      slot.name.c_str());
    return;
  }
  auto request = std::make_shared<StopRequest::Request>();
  request->name = slot.name;
  stop_->async_send_request(request);
}

void PerceptionOrchestrator::request_objects(Slot & slot, bool only_tracked)
{
  if (slot.objects_pending || !get_objects_->service_is_ready()) {
    return;
  }
  auto request = std::make_shared<GetObjects::Request>();
  request->request = slot.name;
  request->only_tracked = only_tracked;
  slot.objects_pending = true;
  const std::string name = slot.name;
  get_objects_->async_send_request(
    request, [this, name](rclcpp::Client<GetObjects>::SharedFuture future) {
      auto & slot = slots_.at(name);
      slot.objects_pending = false;
      if (slot.busy) {
        slot.objects = future.get();
      }
    });
}

// ---------------------------------------------------------------------------- the three tasks

void PerceptionOrchestrator::start_track_object(
  Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<TrackObject>> handle)
{
  const auto goal = handle->get_goal();
  slot.canceling = [handle]() {return handle->is_canceling();};
  slot.abort = [handle, this, &slot](const std::string & reason) {
      auto result = std::make_shared<TrackObject::Result>();
      result->reason = reason;
      result->id = output(slot, "id");
      handle->abort(result);
    };
  slot.feedback = [handle, this, &slot]() {
      auto feedback = std::make_shared<TrackObject::Feedback>();
      feedback->state = output(slot, "state");
      feedback->id = output(slot, "id");
      if (slot.objects && !feedback->id.empty()) {
        for (const auto & object : slot.objects->objects) {
          if (object.id == feedback->id) {
            feedback->object = object;
          }
        }
      }
      handle->publish_feedback(feedback);
      request_objects(slot, true);
    };
  slot.finish = [handle, this, &slot](BT::NodeStatus status) {
      auto result = std::make_shared<TrackObject::Result>();
      result->id = output(slot, "id");
      if (handle->is_canceling()) {
        result->reason = "cancelled";
        handle->canceled(result);
      } else if (status == BT::NodeStatus::SUCCESS) {
        result->reason = "ended";
        handle->succeed(result);
      } else {
        result->reason = output(slot, "state") == "SEARCHING" ? "not found" : "lost";
        handle->abort(result);
      }
      end(slot);
    };
  start(
    slot, "TrackObject", {
      {"request", slot.name}, {"frame_request", slot.name + "_frame"},
      {"class", goal->class_name}, {"category", goal->category}, {"model", goal->model},
      {"frame_id", goal->frame_id.empty() ? "tracked_" + frame_name(goal->class_name) :
        goal->frame_id},
      {"selection", goal->selection.empty() ? "nearest" : goal->selection},
      {"reacquire", goal->reacquire ? "true" : "false"},
    }, {
      {"find_timeout_ms", milliseconds(goal->find_timeout_s, NO_LIMIT_MS)},
      {"lost_timeout_ms", milliseconds(goal->lost_timeout_s, 2000)},
    }, {"id", "state"});
}

void PerceptionOrchestrator::start_wait_for_object(
  Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<WaitForObject>> handle)
{
  const auto goal = handle->get_goal();
  slot.canceling = [handle]() {return handle->is_canceling();};
  slot.abort = [handle](const std::string & reason) {
      auto result = std::make_shared<WaitForObject::Result>();
      result->reason = reason;
      handle->abort(result);
    };
  slot.feedback = [handle, this, &slot]() {
      auto feedback = std::make_shared<WaitForObject::Feedback>();
      int count = 0;
      std::ignore = blackboard_->get(key(slot, "count"), count);
      feedback->count = count;
      handle->publish_feedback(feedback);
    };
  slot.finish = [handle, this, &slot](BT::NodeStatus status) {
      auto result = std::make_shared<WaitForObject::Result>();
      if (handle->is_canceling()) {
        result->reason = "cancelled";
        handle->canceled(result);
        end(slot);
      } else if (status != BT::NodeStatus::SUCCESS) {
        result->reason = "timeout";
        handle->abort(result);
        end(slot);
      } else {
        // The objects of the image in which they were found, then the result
        slot.ending = true;
        slot.objects_pending = false;
        auto request = std::make_shared<GetObjects::Request>();
        request->request = slot.name;
        const std::string name = slot.name;
        get_objects_->async_send_request(
          request, [this, name, handle](rclcpp::Client<GetObjects>::SharedFuture future) {
            auto & slot = slots_.at(name);
            auto result = std::make_shared<WaitForObject::Result>();
            result->found = true;
            result->reason = "found";
            result->objects = future.get()->objects;
            handle->succeed(result);
            end(slot);
          });
      }
    };
  start(
    slot, "WaitForObject", {
      {"request", slot.name}, {"class", goal->class_name}, {"category", goal->category},
      {"model", goal->model}, {"min_count", std::to_string(std::max(1, goal->min_count))},
    }, {{"timeout_ms", milliseconds(goal->timeout_s, NO_LIMIT_MS)}}, {}, {"count"});
}

void PerceptionOrchestrator::start_observe_class(
  Slot & slot, std::shared_ptr<rclcpp_action::ServerGoalHandle<ObserveClass>> handle)
{
  const auto goal = handle->get_goal();
  std::string labels;
  for (const auto & label : goal->labels) {
    labels += (labels.empty() ? "" : "; ") + label;
  }
  std::string prefix = goal->frame_prefix;
  if (prefix.empty()) {
    prefix = goal->labels.empty() ? frame_name(goal->class_name) : "object";
  }
  slot.canceling = [handle]() {return handle->is_canceling();};
  slot.abort = [handle](const std::string & reason) {
      auto result = std::make_shared<ObserveClass::Result>();
      result->reason = reason;
      handle->abort(result);
    };
  slot.feedback = [handle, this, &slot]() {
      auto feedback = std::make_shared<ObserveClass::Feedback>();
      if (slot.objects) {
        feedback->objects = slot.objects->objects;
      }
      handle->publish_feedback(feedback);
      request_objects(slot, true);
    };
  slot.finish = [handle, this, &slot](BT::NodeStatus status) {
      auto result = std::make_shared<ObserveClass::Result>();
      if (handle->is_canceling()) {
        result->reason = "cancelled";
        handle->canceled(result);
      } else {
        result->reason = status == BT::NodeStatus::SUCCESS ? "ended" :
          "the perception system refused a request";
        handle->abort(result);
      }
      end(slot);
    };
  start(
    slot, "ObserveClass", {
      {"request", slot.name}, {"frame_request", slot.name + "_frames"},
      {"class", goal->labels.empty() ? goal->class_name : ""}, {"category", goal->category},
      {"labels", labels}, {"model", goal->model}, {"frame_prefix", prefix},
    }, {}, {});
}

// ---------------------------------------------------------------------------- the FSM

void PerceptionOrchestrator::control_cycle()
{
  if (state_ == State::STARTING_PERCEPTION) {
    manage_perception_system();
    return;
  }
  const auto period = std::chrono::milliseconds(get_parameter("feedback_period_ms").as_int());
  for (auto & [name, slot] : slots_) {
    if (!slot.busy || slot.ending) {
      continue;
    }
    if (slot.canceling && slot.canceling()) {
      // Cancelled: the tree is halted, the result says so
      slot.finish(BT::NodeStatus::FAILURE);
      continue;
    }
    auto runner = std::dynamic_pointer_cast<koch2::BehaviorRunner>(
      runners_.at(name));
    const auto status = runner->get_bt_status();
    if (status == BT::NodeStatus::SUCCESS || status == BT::NodeStatus::FAILURE) {
      RCLCPP_INFO(get_logger(), "%s: %s", name.c_str(), BT::toStr(status).c_str());
      slot.finish(status);
    } else if (now() - slot.last_feedback >= rclcpp::Duration(period)) {
      slot.last_feedback = now();
      slot.feedback();
    }
  }
}

void PerceptionOrchestrator::manage_perception_system()
{
  // Activates /perception_system in cascade: configure and activate it if needed, then accept
  // goals. Asynchronous: the answers arrive in later cycles
  if (perception_call_pending_ || now() - last_perception_check_ < rclcpp::Duration(1s)) {
    return;
  }
  last_perception_check_ = now();
  if (!perception_get_state_->service_is_ready()) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000, "Waiting for %s", system_.c_str());
    return;
  }
  perception_call_pending_ = true;
  perception_get_state_->async_send_request(
    std::make_shared<lifecycle_msgs::srv::GetState::Request>(),
    [this](rclcpp::Client<lifecycle_msgs::srv::GetState>::SharedFuture future) {
      using lifecycle_msgs::msg::State;
      using lifecycle_msgs::msg::Transition;
      const auto id = future.get()->current_state.id;
      perception_call_pending_ = false;
      if (id == State::PRIMARY_STATE_ACTIVE) {
        go_to_state(static_cast<int>(State::PRIMARY_STATE_ACTIVE));
        return;
      }
      if (!get_parameter("manage_perception_system").as_bool()) {
        return;
      }
      auto request = std::make_shared<lifecycle_msgs::srv::ChangeState::Request>();
      if (id == State::PRIMARY_STATE_UNCONFIGURED) {
        request->transition.id = Transition::TRANSITION_CONFIGURE;
      } else if (id == State::PRIMARY_STATE_INACTIVE) {
        request->transition.id = Transition::TRANSITION_ACTIVATE;
      } else {
        return;   // in a transition: asked again in a second
      }
      RCLCPP_INFO(get_logger(), "%s: %s", system_.c_str(),
        request->transition.id == Transition::TRANSITION_CONFIGURE ? "configure" : "activate");
      perception_call_pending_ = true;
      perception_change_state_->async_send_request(
        request, [this](rclcpp::Client<lifecycle_msgs::srv::ChangeState>::SharedFuture) {
          perception_call_pending_ = false;
          last_perception_check_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
        });
    });
}

void PerceptionOrchestrator::go_to_state(int state)
{
  if (state == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    state_ = State::READY;
    RCLCPP_INFO(get_logger(), "%s is active: the perception tasks are served",
      system_.c_str());
  } else {
    state_ = State::STARTING_PERCEPTION;
  }
}

PerceptionOrchestrator::CallbackReturn
PerceptionOrchestrator::on_activate(const rclcpp_lifecycle::State & previous_state)
{
  active_ = true;
  state_ = State::STARTING_PERCEPTION;
  last_perception_check_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
  return BaseOrchestrator::on_activate(previous_state);
}

PerceptionOrchestrator::CallbackReturn
PerceptionOrchestrator::on_deactivate(const rclcpp_lifecycle::State & previous_state)
{
  active_ = false;
  for (auto & [name, slot] : slots_) {
    if (slot.busy) {
      if (slot.abort) {
        slot.abort("the perception orchestrator was deactivated");
      }
      end(slot);
    }
  }
  return BaseOrchestrator::on_deactivate(previous_state);
}

// Registers the type "perception_orchestrator" when mission_executor loads this library
static koch2::OrchestratorRegistrar<PerceptionOrchestrator> registrar(
  "perception_orchestrator");

}  // namespace perception_orchestrator
