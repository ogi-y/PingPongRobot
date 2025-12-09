#include <chrono>
#include <memory>
#include <cmath>
#include <string>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "pingpong_msgs/msg/shot_params.hpp"
#include "pingpong_msgs/srv/target_shot.hpp"

// 定数
constexpr float GRAVITY = 9.81f;

using namespace std::chrono_literals;

class BallisticsNode : public rclcpp::Node
{
public:
    BallisticsNode()
    : Node("ballistics_node")
    {
        this->declare_parameter("robot.x_position", 762.5);
        this->declare_parameter("robot.y_position", 0.0);

        // モータ指令を送るPublisher
        shot_pub_ = this->create_publisher<pingpong_msgs::msg::ShotParams>("shot_command", 10);

        // 戦略から指令を受けるService
        service_ = this->create_service<pingpong_msgs::srv::TargetShot>(
            "target_shot", 
            std::bind(&BallisticsNode::solve_trajectory, this, std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Ballistics Calculator Ready. Send target coordinates!");
    }

private:
    void solve_trajectory(
        const std::shared_ptr<pingpong_msgs::srv::TargetShot::Request> request,
        std::shared_ptr<pingpong_msgs::srv::TargetShot::Response> response)
    {
        float robot_x = this->get_parameter("robot.x_position").as_double();
        float robot_y = this->get_parameter("robot.y_position").as_double();

        // 1. 目標との相対距離を計算
        float dx = request->target_x - robot_x;
        float dy = request->target_y - robot_y;
        float dist_xy = std::sqrt(dx*dx + dy*dy); // 平面距離

        // 2. Yaw (左右角度) の計算
        // atan2(x, y) で角度が出ます。正面Y軸基準なら atan2(dx, dy) 
        // ※ロボットの座標系定義によりますが、ここでは正面0度、左プラスと仮定
        float yaw_rad = std::atan2(-dx, dy); // 座標系に合わせて符号調整
        float yaw_deg = yaw_rad * 180.0f / M_PI;

        // 3. Pitch (上下角度) と Power の計算
        // 本来は物理方程式を解きますが、ここでは簡易的な「距離→パワー変換モデル」を使います
        
        float pitch_deg = 0.0f;
        float base_power = 0.0f;

        // スピードモードによる打ち分け
        // if (request->speed_mode == "fast") {
        //     // 直線的な軌道 (低弾道・高パワー)
        //     pitch_deg = -40.0f; 
        //     base_power = 30.0f + (dist_xy / 2000.0f) * 30.0f;
        
        // } else if (request->speed_mode == "slow") {
        //     // 山なり軌道 (高弾道・低パワー)
        //     pitch_deg = -10.0f; 
        //     base_power = 20.0f + (dist_xy / 2000.0f) * 10.0f;
        
        // } else { // normal
        //     pitch_deg = -20.0f;
        //     base_power = 25.0f + (dist_xy / 2000.0f) * 20.0f;
        // }

        // 奥行きで打ち分け
        if (dist_xy < 1000.0f) {
            base_power = 10;
            pitch_deg = -20.0f; 
        } else if (dist_xy < 3000.0f) {
            base_power = 20;
            pitch_deg = -15.0f;
        }
        else {
            base_power = 30;
            pitch_deg = -10.0f;
        }

        // 4. モータ左右差の計算 (カーブさせたい場合など)
        
        int power_L = static_cast<int>(base_power) + request->spin;
        int power_R = static_cast<int>(base_power) - request->spin;

        // クリップ処理 (-100~100)
        power_L = std::max(-100, std::min(100, power_L));
        power_R = std::max(-100, std::min(100, power_R));

        // 5. メッセージ作成と送信
        auto msg = pingpong_msgs::msg::ShotParams();
        msg.pos = robot_x;
        msg.roll_deg = request->roll;
        msg.pitch_deg = pitch_deg;
        msg.yaw_deg = yaw_deg;
        msg.pow_left = power_L;
        msg.pow_right = power_R;

        shot_pub_->publish(msg);

        // ログとレスポンス
        RCLCPP_INFO(this->get_logger(), 
            "Solved: Tgt(%.0f, %.0f) -> Roll: %.1f, Yaw: %.1f, Pitch: %.1f, PowL: %d, PowR: %d",
            request->target_x, request->target_y, request->roll, yaw_deg, pitch_deg, power_L, power_R);

        response->success = true;
        response->message = "Fired";
    }

    rclcpp::Publisher<pingpong_msgs::msg::ShotParams>::SharedPtr shot_pub_;
    rclcpp::Service<pingpong_msgs::srv::TargetShot>::SharedPtr service_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BallisticsNode>());
    rclcpp::shutdown();
    return 0;
}