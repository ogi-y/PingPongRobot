import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

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
            phi_list = np.linspace(phi-np.pi/4, phi, phi_step)
        else:
            phi_list = np.linspace(phi, phi+np.pi/4, phi_step)
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

# 描画関係
def draw_table_3d(ax):
    X = [-TABLE_WIDTH/2, TABLE_WIDTH/2, TABLE_WIDTH/2, -TABLE_WIDTH/2, -TABLE_WIDTH/2]
    Y = [0, 0, TABLE_LENGTH, TABLE_LENGTH, 0]
    Z = [0, 0, 0, 0, 0]
    ax.plot(X, Y, Z, color='lightblue', lw=2)
    ax.plot_surface(
        np.array([[-TABLE_WIDTH/2, TABLE_WIDTH/2], [-TABLE_WIDTH/2, TABLE_WIDTH/2]]),
        np.array([[0, 0], [TABLE_LENGTH, TABLE_LENGTH]]),
        np.zeros((2, 2)),
        color='lightblue', alpha=0.2
    )
    ax.plot([-TABLE_WIDTH/2, TABLE_WIDTH/2], [NET_Y, NET_Y], [NET_HEIGHT, NET_HEIGHT], 'k', lw=2)
    ax.plot([-TABLE_WIDTH/2, TABLE_WIDTH/2], [NET_Y, NET_Y], [0, 0], 'k', lw=1)
    ax.plot([0, 0], [NET_Y, NET_Y], [0, NET_HEIGHT], 'k', lw=2)

def plot_3views(trajectories):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    draw_table_3d(ax3d)
    for idx, tr in enumerate(trajectories, 1):
        ax3d.plot(tr["x"], tr["y"], tr["z"], label=f"traj {idx}")
        for j, (xb, yb) in enumerate(tr["bounces"]):
            ax3d.scatter(xb, yb, 0, color='r' if j == 0 else 'g', s=40)
    ax3d.set_xlabel("x [m] (Left-Right)")
    ax3d.set_ylabel("y [m] (Front-Back)")
    ax3d.set_zlabel("z [m] (Height)")
    ax3d.set_xlim(-TABLE_WIDTH/2, TABLE_WIDTH/2)
    ax3d.set_ylim(0, TABLE_LENGTH)
    ax3d.set_zlim(0, 1.5)
    ax3d.view_init(elev=20, azim=-60)
    ax3d.set_title("3D View")

    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.add_patch(plt.Rectangle((-TABLE_WIDTH/2, 0), TABLE_WIDTH, TABLE_LENGTH,
                                   color='lightblue', alpha=0.3))
    ax_top.axhline(NET_Y, color='k', linestyle='--', label="Net")
    for idx, tr in enumerate(trajectories, 1):
        ax_top.plot(tr["x"], tr["y"], label=f"traj {idx}")
        for j, (xb, yb) in enumerate(tr["bounces"]):
            ax_top.plot(xb, yb, 'ro' if j == 0 else 'go')
    ax_top.set_xlabel("x [m] (Left-Right)")
    ax_top.set_ylabel("y [m] (Front-Back)")
    ax_top.set_xlim(-TABLE_WIDTH/2, TABLE_WIDTH/2)
    ax_top.set_ylim(0, TABLE_LENGTH)
    ax_top.set_title("Top View (x-y)")
    ax_top.legend()

    ax_side = fig.add_subplot(gs[0, 2])
    ax_side.add_patch(plt.Rectangle((0, 0), TABLE_LENGTH, 0.02,
                                    color='lightblue', alpha=0.4))
    ax_side.axvline(NET_Y, color='k', linestyle='--', label="Net")
    ax_side.axhline(NET_HEIGHT, color='gray', linestyle=':')
    for idx, tr in enumerate(trajectories, 1):
        ax_side.plot(tr["y"], tr["z"], label=f"traj {idx}")
        for j, (xb, yb) in enumerate(tr["bounces"]):
            ax_side.plot(yb, 0, 'ro' if j == 0 else 'go')
    ax_side.set_xlabel("y [m] (Front-Back)")
    ax_side.set_ylabel("z [m] (Height)")
    ax_side.set_xlim(0, TABLE_LENGTH)
    ax_side.set_ylim(0, 1.5)
    ax_side.set_title("Side View (y-z)")
    ax_side.legend()

    plt.tight_layout()
    plt.show()

def main():
    """メモ
    発射高さ27cm,射出速度3m/s程度
    座標系x:左右, y:前後, z:高さ"""
    robo_pos = (-0.5, 0, 0.27) # ロボットの位置
    target_my = None     # 1バウンド目（自分側）
    target_oppo = (-0.5, 2.0)  # 2バウンド目（相手側）
    spin_rate = 300.0  # 回転速度 [rad/s]
    spin_ang = 90 #0:トップ， 90:右サイド，180:バック，270:左サイド
    mode = "speed"
    target_speed = 10.0  # 目標速度 [m/s]
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
    v_list_coast = np.linspace(1.0, 5.0, 5)
    theta_list_coast = np.deg2rad(np.linspace(-45, 45, 5))
    phi_step_coast = 5

    start = time.time()
    coast_v, coast_theta, coast_phi, score = find_best_serve_params(v_list_coast, theta_list_coast, phi_step_coast, robo_pos, target_my, target_oppo, mode=mode, target=target_speed, **sim_kwargs_coast)
    time_coast = time.time() - start
    print("粗探索時間:", time_coast, "秒")

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

    end = time.time()
    print("探索時間:", end - start, "秒")
    print("最適速度:", best_v, "m/s")
    print("最適仰角:", np.rad2deg(best_theta), "度")
    print("最適横回転角:", np.rad2deg(best_phi), "度")
    print("評価スコア:", score)
    x, y, z, b = simulate_trajectory(best_v, best_theta, best_phi, robo_pos, **sim_kwargs_fine)
    plot_3views([{"x": x, "y": y, "z": z, "bounces": b}])

if __name__ == "__main__":
    main()
