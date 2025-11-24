#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>

#include "rclcpp/rclcpp.hpp"
#include "pingpong_msgs/msg/shot_params.hpp"
#include "pingpong_msgs/srv/shoot.hpp" // ★変更: 作成したカスタムサービス

using namespace std::chrono_literals;
constexpr float WIDTH = 2000.0f;

// テンプレート構造体
struct ShotTemplate {
    std::string name;
    int difficulty; 
    float pos;
    float roll;
    float pitch;
    float yaw;
    int8_t pow_right;
    int8_t pow_left;
};

class ShotController : public rclcpp::Node
{
public:
    ShotController()
    : Node("shot_controller")
    {
        publisher_ = this->create_publisher<pingpong_msgs::msg::ShotParams>("shot_command", 10);

        // --- テンプレート定義 (省略なし) ---
        templates_ = {
            {"Basic", 1, WIDTH *0.5, 0.0f, 0.0f, 0.0f, 10, 10},
            {"Basic Left", 1, WIDTH *0.5, 0.0f, 0.0f, -15.0f, 10, 10},
            {"Basic Right", 1, WIDTH *0.5, 0.0f, 0.0f, 15.0f, 10, 10},
            {"Normal Center",  2, WIDTH *0.5, 0.0f, -10.0f, 0.0f, 20, 20},
            {"Wide Left",      2,  WIDTH *0.3, 0.0f, 0.0f, 45.0f, 50, 50},
            {"Wide Right",     2, WIDTH *0.7, 0.0f, 0.0f, -45.0f, 50, 50},
            {"Pro Smash",      3, WIDTH *0.5, 0.0f, -20.0f, 0.0f, 90, 90},
            {"Crazy Curve",    3,  WIDTH *0.1, 0.0f, -5.0f, 25.0f, 80, 30},
            {"Net Edge",       3, WIDTH *0.9, 0.0f, 0.0f, 0.0f, 15, 15},
            {"Impossible Shot", 4, WIDTH *0.1, 0.0f, 0.0f, 45.0f, -50, 100},
        };

        std::random_device seed_gen;
        engine_ = std::mt19937(seed_gen());

        // ★変更: サービス型を Trigger から Shoot に変更
        // サービス名: "shoot"
        service_ = this->create_service<pingpong_msgs::srv::Shoot>(
            "shoot", 
            std::bind(&ShotController::shoot_callback, this, std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Ready! Service '/shoot' available.");
    }

private:
    // ★変更: コールバック関数
    void shoot_callback(
        const std::shared_ptr<pingpong_msgs::srv::Shoot::Request> request,
        std::shared_ptr<pingpong_msgs::srv::Shoot::Response> response)
    {
        auto message = pingpong_msgs::msg::ShotParams();
        std::string shot_name = "Manual Shot";

        // A. マニュアルモード (difficulty == 0)
        if (request->difficulty == 0) {
            message.pos = request->pos;
            message.roll_deg = request->roll_deg;
            message.pitch_deg = request->pitch_deg;
            message.yaw_deg = request->yaw_deg;
            message.pow_right = request->pow_right;
            message.pow_left = request->pow_left;
            
            response->message = "Fired Manual Shot";
        }
        // B. 自動モード (difficulty > 0)
        else {
            // テンプレート抽出
            std::vector<ShotTemplate> candidates;
            for (const auto& t : templates_) {
                if (t.difficulty == request->difficulty) {
                    candidates.push_back(t);
                }
            }

            if (candidates.empty()) {
                RCLCPP_WARN(this->get_logger(), "No templates found for difficulty: %d", request->difficulty);
                response->success = false;
                response->message = "No templates found";
                return;
            }

            std::uniform_int_distribution<> dist(0, candidates.size() - 1);
            const auto& selected = candidates[dist(engine_)];

            message.pos = selected.pos;
            message.roll_deg = selected.roll;
            message.pitch_deg = selected.pitch;
            message.yaw_deg = selected.yaw;
            message.pow_right = selected.pow_right;
            message.pow_left = selected.pow_left;
            shot_name = selected.name;
            
            response->message = "Fired Auto Shot: " + shot_name;
        }

        // Publish
        publisher_->publish(message);

        RCLCPP_INFO(this->get_logger(), "Published: [%s] (Diff request: %d)", shot_name.c_str(), request->difficulty);
        response->success = true;
    }

    rclcpp::Publisher<pingpong_msgs::msg::ShotParams>::SharedPtr publisher_;
    rclcpp::Service<pingpong_msgs::srv::Shoot>::SharedPtr service_; // ★型変更
    std::vector<ShotTemplate> templates_;
    std::mt19937 engine_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ShotController>());
    rclcpp::shutdown();
    return 0;
}