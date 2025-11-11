import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

g = 9.80665  # [m/s^2]

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_Y = TABLE_LENGTH / 2
NET_HEIGHT = 0.1525

def find_bounce_time(z0, vz, restitution_power=0):
    """
    z = z0 + vz*t - 0.5*g*t^2 = 0 となる時刻を解析的に求める
    restitution_power: バウンド回数（反発係数の累乗）
    """
    a = -0.5 * g
    b = vz * (restitution_power ** restitution_power) if restitution_power > 0 else vz
    c = z0
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    valid_times = [t for t in [t1, t2] if t > 1e-9]
    return min(valid_times) if valid_times else None

def simulate_trajectory_analytical(v, theta, phi, robo_pos, restitution=0.9,
                                   max_bounce=3, y_limit=TABLE_LENGTH*2):
    """
    解析解を使った高速バウンドシミュレーション
    """
    vx0 = v * np.cos(theta) * np.sin(phi)
    vy0 = v * np.cos(theta) * np.cos(phi)
    vz0 = v * np.sin(theta)
    
    x0, y0, z0 = robo_pos
    
    xs, ys, zs = [], [], []
    bounces = []
    
    vx, vy, vz = vx0, vy0, vz0
    x, y, z = x0, y0, z0
    t_total = 0
    
    for bounce_num in range(max_bounce):
        t_bounce = find_bounce_time(z, vz, 0)
        
        if t_bounce is None or t_bounce > 10:
            break
        
        t_end = min(t_bounce, (y_limit - y) / vy if vy > 0 else 10)
        
        t_samples = np.linspace(0, t_end, max(int(t_end * 100), 10))
        
        x_traj = x + vx * t_samples
        y_traj = y + vy * t_samples
        z_traj = z + vz * t_samples - 0.5 * g * t_samples**2
        
        xs.extend(x_traj)
        ys.extend(y_traj)
        zs.extend(z_traj)
        
        x = x + vx * t_bounce
        y = y + vy * t_bounce
        z = 0
        t_total += t_bounce
        
        bounces.append((x, y))
        
        if y >= y_limit or z_traj[-1] > 3.0:
            break
        
        vz = -(vz - g * t_bounce) * restitution
    
    return np.array(xs), np.array(ys), np.array(zs), bounces


def evaluate_serve(v, theta, phi, robo_pos, target_my=None, target_oppo=None, mode="speed", target=None, mode_alpha=0.1):
    x, y, z, bounces = simulate_trajectory_analytical(v, theta, phi, robo_pos)
    BAD_SCORE = -10000.0
    SAFE_HEIGHT = 0.05

    if len(bounces) < 2:
        return BAD_SCORE
    for bx, by in bounces[:2]:
        if bx < -TABLE_WIDTH/2 or bx > TABLE_WIDTH/2:
            return BAD_SCORE
        if by < 0 or by > TABLE_LENGTH:
            return BAD_SCORE

    idx_net = np.argmin(np.abs(y - NET_Y))
    if z[idx_net] < NET_HEIGHT + SAFE_HEIGHT:
        return BAD_SCORE
    
    bounce1 = bounces[0]
    bounce2 = bounces[1]

    err1 = (bounce1[0] - target_my[0])**2 + (bounce1[1] - target_my[1])**2 if target_my else 0.0
    err2 = (bounce2[0] - target_oppo[0])**2 + (bounce2[1] - target_oppo[1])**2 if target_oppo else 0.0

    score = -(err1 + err2)
    if mode == "speed":
        if target is not None:
            score += -abs(v - target) * mode_alpha
    if mode == "angle":
        if target is not None:
            score += -abs(theta - target) * mode_alpha

    return score

def find_best_theta(v, phi, robo_pos, target_my, target_oppo,
                              theta_min_deg=-60, theta_max_deg=60, steps=120, mode=None, target=None):
    best_theta = None
    best_score = -1e9

    thetas = np.linspace(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg), steps)

    for theta in thetas:
        score = evaluate_serve(v, theta, phi, robo_pos, target_my, target_oppo, mode=mode, target=target)
        if score > best_score:
            best_score = score
            best_theta = theta

    return best_theta, best_score

def find_best_serve_params(v_list, robo_pos, target_my, target_oppo, mode="speed", target=10.0):
    best_v = None
    best_theta = None
    best_phi = None
    best_score = -1e9

    if target_oppo is not None:
        phi = np.arctan2(target_oppo[0] - robo_pos[0], target_oppo[1] - robo_pos[1])
    elif target_my is not None:
        phi = np.arctan2(target_my[0] - robo_pos[0], target_my[1] - robo_pos[1])
    else:
        phi = 0.0

    for v in v_list:
        theta, score = find_best_theta(v, phi, robo_pos, target_my, target_oppo,
                                        mode=mode, target=target)
        if score > best_score:
            best_score = score
            best_theta = theta
            best_phi = phi
            best_v = v

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
    ax3d.set_xlabel("x [m] (Left–Right)")
    ax3d.set_ylabel("y [m] (Front–Back)")
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
    ax_top.set_xlabel("x [m] (Left–Right)")
    ax_top.set_ylabel("y [m] (Front–Back)")
    ax_top.set_xlim(-TABLE_WIDTH/2, TABLE_WIDTH/2)
    ax_top.set_ylim(0, TABLE_LENGTH)
    ax_top.set_title("Top View (x–y)")
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
    ax_side.set_xlabel("y [m] (Front–Back)")
    ax_side.set_ylabel("z [m] (Height)")
    ax_side.set_xlim(0, TABLE_LENGTH)
    ax_side.set_ylim(0, 1.5)
    ax_side.set_title("Side View (y–z)")
    ax_side.legend()

    plt.tight_layout()
    plt.show()

def main():
    robo_pos = (0.5, 0, 0.27)
    target_my = None
    target_oppo = (0.5, 2.5)
    v_list = np.arange(2.0, 4.0, 0.1)
    mode = "speed"
    target_speed = 0

    start = time.time()
    best_v, best_theta, best_phi, score = find_best_serve_params(v_list, robo_pos, target_my, target_oppo, mode=mode, target=target_speed)
    end = time.time()
    print("探索時間:", end - start, "秒")
    print("最適速度:", best_v, "m/s")
    print("最適仰角:", np.rad2deg(best_theta), "度")
    print("最適横回転角:", np.rad2deg(best_phi), "度")
    print("評価スコア:", score)
    x, y, z, b = simulate_trajectory_analytical(best_v, best_theta, best_phi, robo_pos)
    plot_3views([{"x": x, "y": y, "z": z, "bounces": b}])

if __name__ == "__main__":
    main()