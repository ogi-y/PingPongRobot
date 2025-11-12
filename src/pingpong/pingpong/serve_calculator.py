import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import math
import numpy as np

g = 9.80665  # [m/s^2]

# 物理パラメータ
TABLE_LENGTH = 2.74   # y方向（前後）
TABLE_WIDTH = 1.525   # x方向（左右）
NET_Y = TABLE_LENGTH / 2
NET_HEIGHT = 0.1525

R_BALL = 0.02           # 半径 [m]（40mm球）
A_CROSS = np.pi * R_BALL**2  # 正面投影面積 [m^2]
M_BALL = 0.0027         # 質量 [kg]（約2.7 g）
RHO_AIR = 1.225         # 空気密度 [kg/m^3]

def spin_ang_to_local_omega(spin_rate, spin_ang):
    ang_rad = np.deg2rad(spin_ang)
    wx = 0
    wy = spin_rate * np.cos(ang_rad)
    wz = spin_rate * -np.sin(ang_rad)
    if abs(wx) < 1e-9:
        wx = 0.0
    if abs(wy) < 1e-9:
        wy = 0.0
    if abs(wz) < 1e-9:
        wz = 0.0

    return (wx, wy, wz)

def local_to_global(omega_local, theta, phi):
    v_dir = np.array([
        np.cos(theta) * np.sin(phi),
        np.cos(theta) * np.cos(phi),
        np.sin(theta)
    ])
    v_dir /= np.linalg.norm(v_dir)

    world_up = np.array([0, 0, 1])
    right = np.cross(world_up, v_dir)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        right = np.array([1, 0, 0])
    else:
        right /= right_norm
    up = np.cross(v_dir, right)

    omega_local_arr = np.array(omega_local)
    omega = (
        v_dir * omega_local_arr[0] +
        right * omega_local_arr[1] +
        up * omega_local_arr[2]
    )

    return omega

def simulate_trajectory(v0, theta, phi, robo_pos,
                        omega_local=(0.0, 0.0, 0.0),
                        C_d=0.5, C_l=0.2, # 抗力・揚力係数
                        m=M_BALL,
                        restitution=0.9, # 法線反発係数
                        spin_restitution=0.9, # 回転保持率
                        tangential_friction=0.9, # 接線摩擦係数
                        max_bounce=3,
                        y_limit=TABLE_LENGTH*2,
                        dt=0.001,
                        max_time=3.0):

    omega = local_to_global(omega_local, theta, phi)
    vx = v0 * np.cos(theta) * np.sin(phi)
    vy = v0 * np.cos(theta) * np.cos(phi)
    vz = v0 * np.sin(theta)
    vel = np.array([vx, vy, vz], dtype=float)
    pos = np.array([robo_pos[0], robo_pos[1], robo_pos[2]], dtype=float)
    omega = np.array(omega, dtype=float)
    xs, ys, zs = [pos[0]], [pos[1]], [pos[2]]
    bounces = []
    t = 0.0

    max_steps = int(max_time / dt)

    for _ in range(max_steps):
        speed = np.linalg.norm(vel)
        if speed < 1e-8:
            v_hat = np.zeros(3)
        else:
            v_hat = vel / speed

        F_drag = -0.5 * RHO_AIR * C_d * A_CROSS * speed**2 * v_hat

        omega_norm = np.linalg.norm(omega)
        if omega_norm < 1e-9 or speed < 1e-9:
            F_magnus = np.zeros(3)
        else:
            F_magnus = RHO_AIR * C_l * A_CROSS * R_BALL * np.cross(omega, vel)

        F_grav = np.array([0.0, 0.0, -m * g])
        F_total = F_drag + F_magnus + F_grav
        acc = F_total / m

        vel = vel + acc * dt
        pos = pos + vel * dt
        t += dt

        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

        if pos[2] <= 0 and vel[2] < 0:
            z_prev = zs[-2]
            v_prev_z = (zs[-1] - zs[-2]) / dt
            if abs(v_prev_z) > 1e-9:
                alpha = z_prev / (z_prev - pos[2])
                alpha = np.clip(alpha, 0.0, 1.0)
                pos[0] = xs[-2] + (pos[0] - xs[-2]) * alpha
                pos[1] = ys[-2] + (pos[1] - ys[-2]) * alpha
                pos[2] = 0.0
            else:
                pos[2] = 0.0

            vel_normal = np.array([0.0, 0.0, vel[2]])
            vel_tangent = vel - vel_normal
            vel_normal[2] = -vel_normal[2] * restitution
            vel_tangent = vel_tangent * tangential_friction
            omega = omega * spin_restitution
            vel = vel_tangent + vel_normal

            bounces.append((pos[0], pos[1]))
            # 位置チェック
            if pos[0] < -TABLE_WIDTH/2 or pos[0] > TABLE_WIDTH/2:
                break
            if pos[1] < 0 or pos[1] > TABLE_LENGTH:
                break
            if len(bounces) == 1:
                if pos[1] > NET_Y:
                    break
            else:
                if pos[1] < NET_Y:
                    break
            if len(bounces) >= 2:
                break

        if pos[1] >= y_limit:
            break
        if pos[2] > 5.0:
            break
        if t >= max_time:
            break

    return np.array(xs), np.array(ys), np.array(zs), bounces

def evaluate_serve(v, theta, phi, robo_pos, target_my = None, target_oppo = None, mode = "speed", target = None, **sim_kwargs):
    x, y, z, bounces = simulate_trajectory(v, theta, phi, robo_pos, **sim_kwargs)
    BAD_SCORE = -10000.0
    SAFE_HEIGHT = 0.05  # ネット上を通過するための余裕
    pos_alpha = 100.0 # 位置の重要度
    mode_alpha = 1.0 # モードの重要度

    if len(bounces) < 2:
        return BAD_SCORE
    # simulate内で判定するので多分いらない
    # for bx, by in bounces[:2]:
    #     if bx < -TABLE_WIDTH/2 or bx > TABLE_WIDTH/2:
    #         return BAD_SCORE
    #     if by < 0 or by > TABLE_LENGTH:
    #         return BAD_SCORE

    idx_net = np.argmin(np.abs(y - NET_Y))
    if z[idx_net] < NET_HEIGHT + SAFE_HEIGHT:
        return BAD_SCORE
    bounce1 = bounces[0]
    bounce2 = bounces[1]
    err1 = (bounce1[0] - target_my[0])**2 + (bounce1[1] - target_my[1])**2 if target_my else 0.0
    err2 = (bounce2[0] - target_oppo[0])**2 + (bounce2[1] - target_oppo[1])**2 if target_oppo else 0.0
    score = -(err1 + err2) * pos_alpha
    if mode == "speed":
        if target is not None:
            score += -abs(v - target) * mode_alpha
    if mode == "angle":
        if target is not None:
            score += -abs(theta - target) * mode_alpha

    return score

def find_best_theta(v, theta_list, phi, robo_pos, target_my, target_oppo, mode=None, target=None, **sim_kwargs):
    best_theta = None
    best_score = -1e9

    for theta in theta_list:
        score = evaluate_serve(v, theta, phi, robo_pos, target_my, target_oppo, mode=mode, target=target, **sim_kwargs)
        if score > best_score:
            best_score = score
            best_theta = theta

    return best_theta, best_score

def find_best_serve_params(v_list, theta_list, phi_step, robo_pos, target_my, target_oppo, mode="speed", target=10.0, **sim_kwargs):
    best_v = None
    best_theta = None
    best_phi = None
    best_score = -1e9
    omega_local = sim_kwargs.get("omega_local", (0.0, 0.0, 0.0))
    
    if target_oppo is not None:
        phi = np.arctan2(target_oppo[0] - robo_pos[0], target_oppo[1] - robo_pos[1])
    elif target_my is not None:
        phi = np.arctan2(target_my[0] - robo_pos[0], target_my[1] - robo_pos[1])
    else:
        phi = 0.0

    if abs(omega_local[2]) < 100:
        for v in v_list:
            theta, score = find_best_theta(v, theta_list, phi, robo_pos, target_my, target_oppo,
                                            mode=mode, target=target, **sim_kwargs)
            if score > best_score:
                best_score = score
                best_theta = theta
                best_phi = phi
                best_v = v
    else:
        if omega_local[2] < 0:
            phi_list = np.linspace(phi-np.pi/6, phi, phi_step)
        else:
            phi_list = np.linspace(phi, phi+np.pi/6, phi_step)
        for v in v_list:
            for phi in phi_list:
                theta, score = find_best_theta(v, theta_list, phi, robo_pos, target_my, target_oppo,
                                                mode=mode, target=target, **sim_kwargs)
                if score > best_score:
                    best_score = score
                    best_theta = theta
                    best_phi = phi
                    best_v = v

    return best_v, best_theta, best_phi, best_score

def cal_serve_params(speed, spin, spin_dir, course_x, course_y, serve_pos):
    robo_pos = (serve_pos, 0, 0.27) # ロボットの位置
    target_my = None
    target_oppo = ((course_x-0.5)*TABLE_WIDTH, course_y * (TABLE_LENGTH - NET_Y) + NET_Y)  # 2バウンド目（相手側）
    spin_rate = spin  # 回転速度 [rad/s]
    spin_ang = spin_dir #0:トップ， 90:右サイド，180:バック，270:左サイド
    mode = "speed"
    target_speed = speed  # 目標速度 [m/s]
    omega_local = spin_ang_to_local_omega(spin_rate, spin_ang)

    sim_kwargs_coast = {
        "omega_local": omega_local,  # (wx, wy, wz)
        "C_d": 0.5,
        "C_l": 0.18,
        "m": M_BALL,
        "restitution": 0.9,
        "spin_restitution": 0.85,
        "tangential_friction": 0.9,
        "dt": 0.01,
        "max_time": 1.0
    }
    v_list_coast = np.linspace(1.0, 10.0, 10)
    theta_list_coast = np.deg2rad(np.linspace(-45, 45, 5))
    phi_step_coast = 5

    coast_v, coast_theta, coast_phi, score = find_best_serve_params(v_list_coast, theta_list_coast, phi_step_coast, robo_pos, target_my, target_oppo, mode=mode, target=target_speed, **sim_kwargs_coast)

    sim_kwargs_fine = {
        "omega_local": omega_local,  # (wx, wy, wz)
        "C_d": 0.5,
        "C_l": 0.18,
        "m": M_BALL,
        "restitution": 0.9,
        "spin_restitution": 0.85,
        "tangential_friction": 0.9,
        "dt": 0.005,
        "max_time": 1.0
    }
    v_list_fine = np.linspace(coast_v-0.5, coast_v+0.5, 5)
    theta_list_fine = (np.linspace(coast_theta - np.deg2rad(10), coast_theta + np.deg2rad(10), 5))
    phi_step_fine = 5

    best_v, best_theta, best_phi, score = find_best_serve_params(v_list_fine, theta_list_fine, phi_step_fine, robo_pos, target_my, target_oppo, mode=mode, target=target_speed, **sim_kwargs_fine)

    return best_v, best_theta, best_phi, score

class ServeCalculator(Node):
    def __init__(self):
        super().__init__('serve_calculator')
        self.subscription = self.create_subscription(
            String,
            '/serve_target',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(String, '/serve_params', 10)
        self.get_logger().info('ServeCalculatorノード起動中...')

    def listener_callback(self, msg):
        try:
            serve_target = json.loads(msg.data)
            self.get_logger().info(f"受信: {serve_target}")

            speed = serve_target["speed"] # m/s
            spin = serve_target["spin"] # rad/s
            spin_dir = serve_target["spin_dir"] # 0:トップ， 90:右サイド，180:バック，270:左サイド
            course_x = serve_target["course_x"] # 0~1 (左~右)
            course_y = serve_target["course_y"] # 0~1 (前~後)
            serve_pos = serve_target["serve_pos"] # 0~1 (左~右)
            # 数値に変換
            v, theta, phi, score = cal_serve_params(speed, spin, spin_dir, course_x, course_y, serve_pos)

            serve_param = {
                "robo_pos":serve_pos,
                "roll":int(spin_dir),
                "pitch":int(theta),
                "yaw":int(phi),
                "v":v,
            }

            msg_out = String()
            msg_out.data = json.dumps(serve_param)
            self.publisher.publish(msg_out)
            self.get_logger().info(f"計算結果送信: {serve_param}")

        except Exception as e:
            self.get_logger().error(f"エラー: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ServeCalculator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()