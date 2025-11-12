import numpy as np
import time
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 定数（元コードと同じ） ---
g = 9.80665
TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_Y = TABLE_LENGTH / 2
NET_HEIGHT = 0.1525
R_BALL = 0.02
A_CROSS = np.pi * R_BALL**2
M_BALL = 0.0027
RHO_AIR = 1.225

# --- ヘルパー（ほぼそのまま） ---
def spin_ang_to_local_omega(spin_rate, spin_ang):
    ang = np.deg2rad(spin_ang)
    return (0.0, spin_rate * np.cos(ang), -spin_rate * np.sin(ang))

def local_to_global(omega_local, theta, phi):
    # 方向ベクトル
    ct = np.cos(theta); st = np.sin(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    v_dir = np.array([ct * sp, ct * cp, st])
    # 正規化（theta/phiは正しい範囲前提だが念の為）
    nd = np.sqrt(v_dir[0]*v_dir[0] + v_dir[1]*v_dir[1] + v_dir[2]*v_dir[2])
    if nd == 0.0:
        v_dir = np.array([1.0, 0.0, 0.0])
    else:
        v_dir = v_dir / nd
    world_up = np.array([0.0, 0.0, 1.0])
    # right = cross(world_up, v_dir)
    right = np.array([
        world_up[1]*v_dir[2] - world_up[2]*v_dir[1],
        world_up[2]*v_dir[0] - world_up[0]*v_dir[2],
        world_up[0]*v_dir[1] - world_up[1]*v_dir[0]
    ])
    rn = np.sqrt(right[0]*right[0] + right[1]*right[1] + right[2]*right[2])
    if rn < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rn
    # up = cross(v_dir, right)
    up = np.array([
        v_dir[1]*right[2] - v_dir[2]*right[1],
        v_dir[2]*right[0] - v_dir[0]*right[2],
        v_dir[0]*right[1] - v_dir[1]*right[0]
    ])
    ox, oy, oz = omega_local
    return v_dir * ox + right * oy + up * oz

# --- 高速化した軌道シミュレータ ---
def simulate_trajectory_fast(v0, theta, phi, robo_pos,
                             omega_local=(0.0,0.0,0.0),
                             C_d=0.5, C_l=0.2,
                             m=M_BALL,
                             restitution=0.9,
                             spin_restitution=0.9,
                             tangential_friction=0.9,
                             max_bounce=3,
                             y_limit=TABLE_LENGTH*2,
                             dt=0.001,
                             max_time=3.0):
    # 前処理：局所→世界角速度、初速度ベクトル
    omega = local_to_global(omega_local, theta, phi)
    ct = np.cos(theta); st = np.sin(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    vx = v0 * ct * sp
    vy = v0 * ct * cp
    vz = v0 * st
    vel_x = float(vx); vel_y = float(vy); vel_z = float(vz)
    pos_x = float(robo_pos[0]); pos_y = float(robo_pos[1]); pos_z = float(robo_pos[2])
    ox, oy, oz = float(omega[0]), float(omega[1]), float(omega[2])

    max_steps = int(max_time / dt) + 1
    xs = np.empty(max_steps, dtype=float)
    ys = np.empty(max_steps, dtype=float)
    zs = np.empty(max_steps, dtype=float)
    t = 0.0
    step = 0
    xs[0], ys[0], zs[0] = pos_x, pos_y, pos_z
    bounces = []

    # ループ内定数（乗算だけで済むように）
    half_rho_A = 0.5 * RHO_AIR * A_CROSS
    ball_radius = R_BALL

    while step + 1 < max_steps:
        # speed and hat
        speed_sq = vel_x*vel_x + vel_y*vel_y + vel_z*vel_z
        if speed_sq <= 1e-18:
            v_hat_x = v_hat_y = v_hat_z = 0.0
            speed = 0.0
        else:
            speed = np.sqrt(speed_sq)
            inv_speed = 1.0 / speed
            v_hat_x = vel_x * inv_speed
            v_hat_y = vel_y * inv_speed
            v_hat_z = vel_z * inv_speed

        # drag: -0.5 * rho * C_d * A * speed^2 * v_hat
        drag_factor = -half_rho_A * C_d * speed_sq
        Fdx = drag_factor * v_hat_x
        Fdy = drag_factor * v_hat_y
        Fdz = drag_factor * v_hat_z

        # magnus: rho * C_l * A * R * cross(omega, vel)
        # cross(omega, vel) = (oy*vel_z - oz*vel_y, oz*vel_x - ox*vel_z, ox*vel_y - oy*vel_x)
        if (abs(ox) < 1e-12 and abs(oy) < 1e-12 and abs(oz) < 1e-12) or speed == 0.0:
            Fmx = Fmy = Fmz = 0.0
        else:
            clA_R = RHO_AIR * C_l * A_CROSS * ball_radius
            cx = oy*vel_z - oz*vel_y
            cy = oz*vel_x - ox*vel_z
            cz = ox*vel_y - oy*vel_x
            Fmx = clA_R * cx
            Fmy = clA_R * cy
            Fmz = clA_R * cz

        # gravity
        Fgx = 0.0; Fgy = 0.0; Fgz = -m * g

        # total acceleration
        ax = (Fdx + Fmx + Fgx) / m
        ay = (Fdy + Fmy + Fgy) / m
        az = (Fdz + Fmz + Fgz) / m

        # integrate
        vel_x += ax * dt
        vel_y += ay * dt
        vel_z += az * dt

        pos_x += vel_x * dt
        pos_y += vel_y * dt
        pos_z += vel_z * dt

        step += 1
        xs[step] = pos_x; ys[step] = pos_y; zs[step] = pos_z
        t += dt

        # ground contact
        if pos_z <= 0.0 and vel_z < 0.0:
            # 補間で接地座標を推定（元コードと同様）
            z_prev = zs[step-1]
            if abs(z_prev - pos_z) > 1e-12:
                alpha = z_prev / (z_prev - pos_z)
                if alpha < 0.0: alpha = 0.0
                if alpha > 1.0: alpha = 1.0
                # 補間（x,y）
                pos_x = xs[step-1] + (pos_x - xs[step-1]) * alpha
                pos_y = ys[step-1] + (pos_y - ys[step-1]) * alpha
            pos_z = 0.0

            # 分離速度と接線速度
            vel_normal_z = vel_z
            vel_normal_z = -vel_normal_z * restitution
            vel_tangent_x = vel_x
            vel_tangent_y = vel_y
            # 接線摩擦・スピン保持
            vel_tangent_x *= tangential_friction
            vel_tangent_y *= tangential_friction
            ox *= spin_restitution; oy *= spin_restitution; oz *= spin_restitution
            vel_x = vel_tangent_x
            vel_y = vel_tangent_y
            vel_z = vel_normal_z

            bounces.append((pos_x, pos_y))

            # 位置チェック（元コードロジックを維持）
            if pos_x < -TABLE_WIDTH/2 or pos_x > TABLE_WIDTH/2:
                break
            if pos_y < 0 or pos_y > TABLE_LENGTH:
                break
            if len(bounces) == 1:
                if pos_y > NET_Y:
                    break
            else:
                if pos_y < NET_Y:
                    break
            if len(bounces) >= max_bounce:
                break

        # 早期終了条件
        if pos_y >= y_limit:
            break
        if pos_z > 5.0:
            break
        if t >= max_time:
            break

    # スライスして返す
    xs = xs[:step+1]; ys = ys[:step+1]; zs = zs[:step+1]
    return xs, ys, zs, bounces

# --- 既存の評価・探索関数はそのまま使えるが、simulateを差し替えればOK ---
def evaluate_serve(v, theta, phi, robo_pos, target_my=None, target_oppo=None, mode="speed", target=None, **sim_kwargs):
    xs, ys, zs, bounces = simulate_trajectory_fast(v, theta, phi, robo_pos, **sim_kwargs)
    BAD_SCORE = -10000.0
    SAFE_HEIGHT = 0.05
    pos_alpha = 100.0
    mode_alpha = 1.0

    if len(bounces) < 2:
        return BAD_SCORE
    idx_net = np.argmin(np.abs(ys - NET_Y))
    if zs[idx_net] < NET_HEIGHT + SAFE_HEIGHT:
        return BAD_SCORE
    b1 = bounces[0]; b2 = bounces[1]
    err1 = (b1[0] - target_my[0])**2 + (b1[1] - target_my[1])**2 if target_my else 0.0
    err2 = (b2[0] - target_oppo[0])**2 + (b2[1] - target_oppo[1])**2 if target_oppo else 0.0
    score = -(err1 + err2) * pos_alpha
    if mode == "speed" and target is not None:
        score += -abs(v - target) * mode_alpha
    if mode == "angle" and target is not None:
        score += -abs(theta - target) * mode_alpha
    return score

# --- 並列探索ユーティリティ（v と phi の組み合わせを並列化） ---
def worker_eval(args):
    v, theta_list, phi, robo_pos, target_my, target_oppo, mode, target, sim_kwargs = args
    best_theta = None; best_score = -1e9
    for th in theta_list:
        s = evaluate_serve(v, th, phi, robo_pos, target_my=target_my, target_oppo=target_oppo, mode=mode, target=target, **sim_kwargs)
        if s > best_score:
            best_score = s; best_theta = th
    return (v, best_theta, phi, best_score)

def find_best_serve_params_parallel(v_list, theta_list, phi_step, robo_pos, target_my, target_oppo, mode="speed", target=10.0, **sim_kwargs):
    omega_local = sim_kwargs.get("omega_local", (0.0,0.0,0.0))
    if target_oppo is not None:
        base_phi = np.arctan2(target_oppo[0] - robo_pos[0], target_oppo[1] - robo_pos[1])
    elif target_my is not None:
        base_phi = np.arctan2(target_my[0] - robo_pos[0], target_my[1] - robo_pos[1])
    else:
        base_phi = 0.0

    args = []
    if abs(omega_local[2]) < 100:
        phi_list = [base_phi]
    else:
        if omega_local[2] < 0:
            phi_list = list(np.linspace(base_phi - np.pi/4, base_phi, phi_step))
        else:
            phi_list = list(np.linspace(base_phi, base_phi + np.pi/4, phi_step))

    # ワーカー引数一覧を作る
    for v in v_list:
        for phi in phi_list:
            args.append((v, theta_list, phi, robo_pos, target_my, target_oppo, mode, target, sim_kwargs))

    # 並列実行
    nproc = max(1, cpu_count() - 1)
    with Pool(nproc) as p:
        results = p.map(worker_eval, args)

    # 結果を集計
    best_score = -1e9
    best_v = best_theta = best_phi = None
    for v, th, phi, s in results:
        if s > best_score:
            best_score = s; best_v = v; best_theta = th; best_phi = phi
    return best_v, best_theta, best_phi, best_score

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

# ---------------------------
# 使い方（例）
if __name__ == "__main__":
    robo_pos = (-0.5, 0.0, 0.27)
    target_my = None
    target_oppo = (-0.5, 2.0)
    omega_local = spin_ang_to_local_omega(500.0, -90)
    sim_kwargs = {
        "omega_local": omega_local,
        "C_d": 0.5,
        "C_l": 0.18,
        "m": M_BALL,
        "restitution": 0.9,
        "spin_restitution": 0.85,
        "tangential_friction": 0.9,
        "dt": 0.005,
        "max_time": 1.0
    }
    v_list = np.linspace(1.0, 5.0, 10)
    theta_list = np.deg2rad(np.linspace(-45, 45, 15))
    start = time.time()
    best_v, best_theta, best_phi, best_score = find_best_serve_params_parallel(v_list, theta_list, 5, robo_pos, target_my, target_oppo, mode="speed", target=10.0, **sim_kwargs)
    print("time:", time.time() - start, "best:", best_v, np.rad2deg(best_theta), np.rad2deg(best_phi), best_score)
    x, y, z, b = simulate_trajectory_fast(best_v, best_theta, best_phi, robo_pos, **sim_kwargs)
    plot_3views([{"x": x, "y": y, "z": z, "bounces": b}])
