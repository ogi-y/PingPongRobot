#include <chrono>
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "pingpong_msgs/msg/shot_params.hpp"
#include "pingpong_msgs/srv/target_shot.hpp"

// --- 物理定数 ---
constexpr double G = 9.80665;
constexpr double RHO = 1.204;
constexpr double MASS = 0.0027;
constexpr double RADIUS = 0.020;
constexpr double AREA = M_PI * RADIUS * RADIUS;
constexpr double Cd = 0.5;
constexpr double Cm = 0.25;

using namespace std::chrono_literals;

struct Vector3 {
    double x, y, z;
    Vector3 operator+(const Vector3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vector3 operator-(const Vector3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vector3 operator*(double s) const { return {x * s, y * s, z * s}; }
};

class BallisticsNode : public rclcpp::Node
{
public:
    BallisticsNode()
    : Node("ballistics_node")
    {
        this->declare_parameter("robot.x_position", 762.5);
        this->declare_parameter("robot.y_position", 0.0);

        shot_pub_ = this->create_publisher<pingpong_msgs::msg::ShotParams>("shot_command", 10);

        service_ = this->create_service<pingpong_msgs::srv::TargetShot>(
            "target_shot", 
            std::bind(&BallisticsNode::solve_trajectory, this, std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Physics Engine Ready.");
    }

private:
    double power_to_v0(double avg_power) {
        return 2.0 + (std::abs(avg_power) / 100.0) * 23.0; 
    }

    double spin_to_omega(double spin_val) {
        return (spin_val / 100.0) * 300.0;
    }

    // --- 弾道シミュレーション ---
    Vector3 simulate_shot(double v0, double pitch_deg, double yaw_deg, double omega, double roll_deg) {
        double dt = 0.0001; 
        
        double pitch_rad = pitch_deg * M_PI / 180.0;
        double yaw_rad = yaw_deg * M_PI / 180.0;
        double roll_rad = roll_deg * M_PI / 180.0;

        Vector3 v = {
            v0 * std::cos(pitch_rad) * std::sin(yaw_rad),
            v0 * std::cos(pitch_rad) * std::cos(yaw_rad),
            v0 * std::sin(pitch_rad)
        };

        Vector3 w = { 
            -std::sin(roll_rad) * omega,
            0.0, 
            std::cos(roll_rad) * omega
        }; 

        Vector3 p = {0.0, 0.0, 0.3};
        double t = 0.0;

        while (p.z > 0.0 && t < 3.0) {
            double vel = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
            
            double Fd = 0.5 * Cd * RHO * AREA * vel * vel;
            Vector3 a_drag = v * (-1.0 * Fd / (MASS * vel));

            Vector3 cross = {
                w.y*v.z - w.z*v.y,
                w.z*v.x - w.x*v.z,
                w.x*v.y - w.y*v.x
            };
            double S = 0.5 * Cm * RHO * AREA * RADIUS; 
            Vector3 a_mag = cross * (S / MASS);

            Vector3 a_grav = {0.0, 0.0, -G};

            Vector3 a_total = a_drag + a_mag + a_grav;
            v = v + a_total * dt;
            p = p + v * dt;
            t += dt;
        }
        return p;
    }

    // グリッド探索
    bool solve_angles(double target_rel_x, double target_rel_y, double v0, double omega, double roll,
                      double& best_pitch, double& best_yaw) 
    {
        double min_dist_sq = 1000000.0;
        
        for (double p = -10.0; p <= 45.0; p += 1.0) {
            for (double y = -30.0; y <= 30.0; y += 1.0) {
                
                Vector3 land_pos = simulate_shot(v0, p, y, omega, roll);
                
                double dx = land_pos.x - target_rel_x;
                double dy = land_pos.y - target_rel_y;
                double dist_sq = dx*dx + dy*dy;

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_pitch = p;
                    best_yaw = y;
                }
            }
        }
        return (min_dist_sq < 1.0); 
    }

void solve_trajectory(
        const std::shared_ptr<pingpong_msgs::srv::TargetShot::Request> request,
        std::shared_ptr<pingpong_msgs::srv::TargetShot::Response> response)
    {
        float robot_x = this->get_parameter("robot.x_position").as_double();
        float robot_y = this->get_parameter("robot.y_position").as_double();

        double target_rel_x = (request->target_x - robot_x) / 1000.0;
        double target_rel_y = (request->target_y - robot_y) / 1000.0;
        
        double total_speed = (double)request->speed;
        double base_power = total_speed / 2.0;

        double spin_val = (double)request->spin;
        double power_L = base_power + spin_val;
        double power_R = base_power - spin_val;
        
        // 物理パラメータ変換
        double v0 = power_to_v0((power_L + power_R) / 2.0);
        double omega = spin_to_omega(power_L - power_R);
        double roll = request->roll_deg;

        // 計算実行
        double best_pitch = 0.0;
        double best_yaw = 0.0;
        
        bool found = solve_angles(target_rel_x, target_rel_y, v0, omega, roll, best_pitch, best_yaw);

        // 送信
        auto msg = pingpong_msgs::msg::ShotParams();
        msg.pos = robot_x;
        msg.roll_deg = roll;
        msg.pitch_deg = best_pitch;
        msg.yaw_deg = best_yaw; 
        msg.pow_left = (int)power_L;
        msg.pow_right = (int)power_R;

        shot_pub_->publish(msg);

        // ログ
        if (found) {
            RCLCPP_INFO(this->get_logger(), 
                "SUCCESS: Speed=%d(L%d/R%d) -> Pitch=%.1f, Yaw=%.1f",
                (int)total_speed, (int)power_L, (int)power_R, best_pitch, best_yaw);
            response->success = true;
            response->message = "Calculated optimal trajectory";
        } else {
            RCLCPP_WARN(this->get_logger(), 
                "UNREACHABLE: Power %d is too weak/strong for dist %.1fm", (int)total_speed, target_rel_y);
            response->success = false;
            response->message = "Target Unreachable";
        }
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