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

// The Behavior Tree nodes against fake services of the perception system, in one process: the
// plugin is loaded as a behavior would load it.

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "behaviortree_cpp/bt_factory.h"
#include "rclcpp/rclcpp.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"

#include "perception_system_interfaces/srv/detect_once.hpp"
#include "perception_system_interfaces/srv/start_detection.hpp"
#include "perception_system_interfaces/srv/start_target_frame.hpp"
#include "perception_system_interfaces/srv/stop_request.hpp"

using namespace std::chrono_literals;
namespace srv = perception_system_interfaces::srv;

class FakePerceptionSystem
{
public:
  explicit FakePerceptionSystem(rclcpp::Node::SharedPtr node)
  {
    start_ = node->create_service<srv::StartDetection>(
      "/perception_system/start_detection",
      [this](const std::shared_ptr<srv::StartDetection::Request> request,
      std::shared_ptr<srv::StartDetection::Response> response) {
        last_class = request->class_name;
        last_labels = request->labels.size();
        response->success = request->class_name != "refused";
        response->message = response->success ? "ok" : "refused for the test";
        response->detections_2d_topic = "/perception_system/requests/" + request->name +
        "/detections_2d";
      });
    frame_ = node->create_service<srv::StartTargetFrame>(
      "/perception_system/start_target_frame",
      [this](const std::shared_ptr<srv::StartTargetFrame::Request> request,
      std::shared_ptr<srv::StartTargetFrame::Response> response) {
        last_mode = request->mode;
        last_frame = request->frame_id;
        response->success = true;
      });
    stop_ = node->create_service<srv::StopRequest>(
      "/perception_system/stop",
      [this](const std::shared_ptr<srv::StopRequest::Request> request,
      std::shared_ptr<srv::StopRequest::Response> response) {
        stopped = request->name;
        response->success = true;
      });
    once_ = node->create_service<srv::DetectOnce>(
      "/perception_system/detect_once",
      [](const std::shared_ptr<srv::DetectOnce::Request> request,
      std::shared_ptr<srv::DetectOnce::Response> response) {
        response->success = true;
        if (request->class_name == "cup") {
          response->detections_2d.detections.resize(2);
        }
      });
  }

  std::string last_class, last_frame, stopped;
  size_t last_labels = 0;
  int last_mode = -1;

private:
  rclcpp::Service<srv::StartDetection>::SharedPtr start_;
  rclcpp::Service<srv::StartTargetFrame>::SharedPtr frame_;
  rclcpp::Service<srv::StopRequest>::SharedPtr stop_;
  rclcpp::Service<srv::DetectOnce>::SharedPtr once_;
};

class PerceptionNodes : public ::testing::Test
{
protected:
  void SetUp() override
  {
    node_ = rclcpp::Node::make_shared("test_perception_nodes");
    server_node_ = rclcpp::Node::make_shared("fake_perception_system");
    fake_ = std::make_unique<FakePerceptionSystem>(server_node_);
    executor_.add_node(server_node_);
    executor_.add_node(node_);
    spinner_ = std::thread([this]() {executor_.spin();});
    factory_.registerFromPlugin(PLUGIN_PATH);
  }

  void TearDown() override
  {
    executor_.cancel();
    spinner_.join();
  }

  BT::NodeStatus run(const std::string & body, BT::Blackboard::Ptr blackboard = nullptr,
    std::function<void()> each_tick = nullptr)
  {
    if (!blackboard) {
      blackboard = BT::Blackboard::create();
    }
    blackboard->set("node", node_);
    auto tree = factory_.createTreeFromText(
      "<root BTCPP_format=\"4\"><BehaviorTree ID=\"T\">" + body + "</BehaviorTree></root>",
      blackboard);
    auto status = BT::NodeStatus::RUNNING;
    for (int i = 0; i < 300 && status == BT::NodeStatus::RUNNING; ++i) {
      if (each_tick) {
        each_tick();
      }
      status = tree.tickOnce();
      std::this_thread::sleep_for(10ms);
    }
    return status;
  }

  rclcpp::Node::SharedPtr node_, server_node_;
  std::unique_ptr<FakePerceptionSystem> fake_;
  rclcpp::executors::MultiThreadedExecutor executor_;
  std::thread spinner_;
  BT::BehaviorTreeFactory factory_;
};

TEST_F(PerceptionNodes, StartDetectionOutputsItsTopics)
{
  auto blackboard = BT::Blackboard::create();
  EXPECT_EQ(
    run(
      "<StartDetection request=\"b\" class=\"cup\" labels=\"a; b ;c\" "
      "detections_2d_topic=\"{topic}\"/>", blackboard), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(fake_->last_class, "cup");
  EXPECT_EQ(fake_->last_labels, 3u);
  EXPECT_EQ(blackboard->get<std::string>("topic"), "/perception_system/requests/b/detections_2d");
}

TEST_F(PerceptionNodes, RefusedRequestFails)
{
  EXPECT_EQ(run("<StartDetection request=\"b\" class=\"refused\"/>"), BT::NodeStatus::FAILURE);
  EXPECT_EQ(run("<StartDetection class=\"cup\"/>"), BT::NodeStatus::FAILURE);   // no request
}

TEST_F(PerceptionNodes, MissingServiceFailsAfterTheTimeout)
{
  EXPECT_EQ(
    run("<StopPerception request=\"b\" service_name=\"/nobody\" timeout_ms=\"200\"/>"),
    BT::NodeStatus::FAILURE);
}

TEST_F(PerceptionNodes, TargetFrameAndStop)
{
  EXPECT_EQ(
    run(
      "<Sequence><StartTargetFrame request=\"f\" detection_request=\"b\" frame_id=\"target\" "
      "class=\"person\" mode=\"group\" n_detections=\"2\"/>"
      "<StopPerception request=\"f\"/></Sequence>"), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(fake_->last_mode, srv::StartTargetFrame::Request::MODE_GROUP);
  EXPECT_EQ(fake_->last_frame, "target");
  EXPECT_EQ(fake_->stopped, "f");
}

TEST_F(PerceptionNodes, DetectOnceCounts)
{
  auto blackboard = BT::Blackboard::create();
  EXPECT_EQ(run("<DetectOnce class=\"cup\" count=\"{n}\"/>", blackboard), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<int>("n"), 2);
  EXPECT_EQ(run("<DetectOnce class=\"cup\" min_count=\"3\"/>"), BT::NodeStatus::FAILURE);
  EXPECT_EQ(run("<DetectOnce class=\"bowl\"/>"), BT::NodeStatus::FAILURE);
}

TEST_F(PerceptionNodes, DetectionSeenReadsTheLatestMessage)
{
  auto publisher = server_node_->create_publisher<vision_msgs::msg::Detection2DArray>(
    "/test/detections_2d", 10);
  vision_msgs::msg::Detection2DArray msg;
  msg.detections.resize(1);
  msg.detections[0].id = "4";
  msg.detections[0].results.resize(1);
  msg.detections[0].results[0].hypothesis.class_id = "person";
  auto publish = [&]() {publisher->publish(msg);};
  const std::string retry = "<RetryUntilSuccessful num_attempts=\"100\"><Sequence>"
    "<Sleep msec=\"10\"/>";
  EXPECT_EQ(
    run(retry + "<DetectionSeen topic=\"/test/detections_2d\" class=\"person\" id=\"4\"/>"
    "</Sequence></RetryUntilSuccessful>", nullptr, publish), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(
    run(
      "<RetryUntilSuccessful num_attempts=\"20\"><Sequence><Sleep msec=\"10\"/>"
      "<DetectionSeen topic=\"/test/detections_2d\" class=\"person\" id=\"5\"/>"
      "</Sequence></RetryUntilSuccessful>", nullptr, publish), BT::NodeStatus::FAILURE);
}

TEST_F(PerceptionNodes, CountsAndSelectsDetections)
{
  auto publisher_2d = server_node_->create_publisher<vision_msgs::msg::Detection2DArray>(
    "/test/count_2d", 10);
  auto publisher_3d = server_node_->create_publisher<vision_msgs::msg::Detection3DArray>(
    "/test/select_3d", 10);
  vision_msgs::msg::Detection2DArray msg_2d;
  vision_msgs::msg::Detection3DArray msg_3d;
  const std::vector<std::tuple<std::string, std::string, double>> detections = {
    {"7", "person", 4.0}, {"", "person", 1.0}, {"9", "person", 2.0}, {"3", "dog", 0.5}};
  for (const auto & [id, class_name, z] : detections) {
    vision_msgs::msg::Detection2D d2;
    d2.id = id;
    d2.results.resize(1);
    d2.results[0].hypothesis.class_id = class_name;
    msg_2d.detections.push_back(d2);
    vision_msgs::msg::Detection3D d3;
    d3.id = id;
    d3.results.resize(1);
    d3.results[0].hypothesis.class_id = class_name;
    d3.bbox.center.position.z = z;
    msg_3d.detections.push_back(d3);
  }
  auto publish = [&]() {publisher_2d->publish(msg_2d); publisher_3d->publish(msg_3d);};
  auto retry = [](const std::string & node) {
      return "<RetryUntilSuccessful num_attempts=\"100\"><Sequence><Sleep msec=\"10\"/>" + node +
             "</Sequence></RetryUntilSuccessful>";
    };
  auto blackboard = BT::Blackboard::create();
  EXPECT_EQ(
    run(retry("<DetectionSeen topic=\"/test/count_2d\" class=\"person\" min_count=\"3\" "
      "count=\"{n}\"/>"), blackboard, publish), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<int>("n"), 3);
  // The nearest confirmed person (the one without an id is not a candidate)
  EXPECT_EQ(
    run(retry("<SelectDetection topic=\"/test/select_3d\" class=\"person\" id=\"{id}\"/>"),
    blackboard, publish), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<std::string>("id"), "9");
  EXPECT_EQ(
    run(retry("<SelectDetection topic=\"/test/select_3d\" class=\"person\" policy=\"first\" "
      "id=\"{id}\"/>"), blackboard, publish), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(blackboard->get<std::string>("id"), "7");
}

TEST_F(PerceptionNodes, EachModeNeedsNoClass)
{
  EXPECT_EQ(
    run("<StartTargetFrame request=\"f\" detection_request=\"b\" frame_id=\"obj\" "
      "mode=\"each\"/>"), BT::NodeStatus::SUCCESS);
  EXPECT_EQ(fake_->last_mode, srv::StartTargetFrame::Request::MODE_EACH);
  EXPECT_EQ(
    run("<StartTargetFrame request=\"f\" detection_request=\"b\" frame_id=\"obj\"/>"),
    BT::NodeStatus::FAILURE);   // single without a class
}

TEST_F(PerceptionNodes, PerceptionBehaviorsLoad)
{
  // The three perception behaviors build with the plugin, as a runner of the orchestrator does
  for (const std::string behavior : {"track_object", "wait_for_object", "observe_class"}) {
    EXPECT_NO_THROW(factory_.registerBehaviorTreeFromFile(
        std::string(BEHAVIORS_DIR) + "/" + behavior + ".xml")) << behavior;
  }
  const auto ids = factory_.registeredBehaviorTrees();
  for (const std::string id : {"TrackObject", "WaitForObject", "ObserveClass"}) {
    EXPECT_NE(std::find(ids.begin(), ids.end(), id), ids.end()) << id;
  }
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
