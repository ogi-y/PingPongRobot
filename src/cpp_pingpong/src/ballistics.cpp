#include <chrono>
#include <memory>
#include <cmath>
#include <string>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "pingpong_msgs/msg/shot_params.hpp"
#include "pingpong_msgs/srv/target_shot.hpp"

// 定数
constexpr float ROBOT_X = 1000.0f; // ロボットの設置位置X (中央)
constexpr float ROBOT_Y = 0.0f;    // ロボットの設置位置Y (手前)
constexpr float GRAVITY = 9.81f;

using namespace std::chrono_literals;

class BallisticsNode : public rclcpp::Node
{
public:
    BallisticsNode()
    : Node("ballistics_node")
    {
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
        // 1. 目標との相対距離を計算
        float dx = request->target_x - ROBOT_X;
        float dy = request->target_y - ROBOT_Y;
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
        if (request->speed_mode == "fast") {
            // 直線的な軌道 (低弾道・高パワー)
            pitch_deg = -5.0f; 
            // 距離に応じて線形にパワーを上げる簡易モデル
            // 例: 距離1mでパワー50, 2mでパワー90
            base_power = 30.0f + (dist_xy / 2000.0f) * 60.0f;
        
        } else if (request->speed_mode == "slow") {
            // 山なり軌道 (高弾道・低パワー)
            pitch_deg = 5.0f; 
            base_power = 20.0f + (dist_xy / 2000.0f) * 40.0f;
        
        } else { // normal
            pitch_deg = -2.0f;
            base_power = 25.0f + (dist_xy / 2000.0f) * 50.0f;
        }

        // 4. モータ左右差の計算 (カーブさせたい場合など)
        // 今回はシンプルに左右同じ
        int power_L = static_cast<int>(base_power);
        int power_R = static_cast<int>(base_power);

        // クリップ処理 (0~100)
        power_L = std::max(0, std::min(100, power_L));
        power_R = std::max(0, std::min(100, power_R));

        // 5. メッセージ作成と送信
        auto msg = pingpong_msgs::msg::ShotParams();
        msg.pos = ROBOT_X; // ロボット自体の位置は固定
        msg.roll_deg = 0.0f;
        msg.pitch_deg = pitch_deg;
        msg.yaw_deg = yaw_deg;
        msg.pow_left = power_L;
        msg.pow_right = power_R;

        shot_pub_->publish(msg);

        // ログとレスポンス
        RCLCPP_INFO(this->get_logger(), 
            "Solved: Tgt(%.0f, %.0f) -> Yaw:%.1f, Pitch:%.1f, Pwr:%d",
            request->target_x, request->target_y, yaw_deg, pitch_deg, power_L);

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