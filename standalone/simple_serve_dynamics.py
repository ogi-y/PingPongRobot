import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

g = 9.80665  # [m/s^2]

# 卓球台パラメータ
TABLE_LENGTH = 2.74   # y方向（前後）
TABLE_WIDTH = 1.525   # x方向（左右）
NET_Y = TABLE_LENGTH / 2
NET_HEIGHT = 0.1525

# ========== 軌道シミュレーション ==========
def simulate_trajectory(v, theta, phi, robo_pos, restitution=0.9,
                        max_bounce=3, y_limit=TABLE_LENGTH*2, dt=0.001):
    """
    バウンドを考慮したシンプル放物運動シミュレーション

    v: 初速 [m/s]
    theta: 仰角 [rad]
    phi: 左右方向角 [rad] (右が正)
    restitution: 反発係数（0〜1）
    max_bounce: 最大バウンド回数
    y_limit: シミュレーションを終了する y距離
    """

    # 初期速度成分
    vx = v * np.cos(theta) * np.sin(phi)
    vy = v * np.cos(theta) * np.cos(phi)
    vz = v * np.sin(theta)

    # 現在位置・速度
    x, y, z = robo_pos[0], robo_pos[1], robo_pos[2]
    t = 0.0

    xs, ys, zs = [x], [y], [z]
    bounces = []

    for _ in range(int(10 / dt)):  # 安全上限
        # 更新
        x += vx * dt
        y += vy * dt
        z += vz * dt - 0.5 * g * dt**2
        vz -= g * dt
        t += dt

        xs.append(x)
        ys.append(y)
        zs.append(z)

        # 台に到達したらバウンド
        if z <= 0 and vz < 0:
            z = 0
            vz = -vz * restitution  # 反発
            bounces.append((x, y))
            if len(bounces) >= max_bounce:
                break

        # y方向の距離制限で停止
        if y >= y_limit:
            break

        # 高すぎる場合も停止
        if z > 3.0:
            break

    return np.array(xs), np.array(ys), np.array(zs), bounces


def evaluate_serve(v, theta, phi, robo_pos, target_my = None, target_oppo = None, mode = "speed", target = None, mode_alpha=1.0):
    """
    サーブの評価関数

    target_my: 自分側の目標座標 (x, y)
    target_oppo: 相手側の目標座標 (x, y)
    """
    x, y, z, bounces = simulate_trajectory(v, theta, phi, robo_pos)
    BAD_SCORE = -10000.0
    SAFE_HEIGHT = 0.05  # ネット上を通過するための余裕

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
                              theta_min_deg=-45, theta_max_deg=45, steps=90, mode=None, target=None):
    """
    速度 v・左右角 phi・狙い座標を固定して
    最適な仰角 theta を探索する
    """
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

    #xy平面上でphiを求める
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

# ========== 卓球台を描画 (3D) ==========
def draw_table_3d(ax):
    # 台の輪郭
    X = [-TABLE_WIDTH/2, TABLE_WIDTH/2, TABLE_WIDTH/2, -TABLE_WIDTH/2, -TABLE_WIDTH/2]
    Y = [0, 0, TABLE_LENGTH, TABLE_LENGTH, 0]
    Z = [0, 0, 0, 0, 0]
    ax.plot(X, Y, Z, color='lightblue', lw=2)
    # 台面
    ax.plot_surface(
        np.array([[-TABLE_WIDTH/2, TABLE_WIDTH/2], [-TABLE_WIDTH/2, TABLE_WIDTH/2]]),
        np.array([[0, 0], [TABLE_LENGTH, TABLE_LENGTH]]),
        np.zeros((2, 2)),
        color='lightblue', alpha=0.2
    )
    # ネット
    ax.plot([-TABLE_WIDTH/2, TABLE_WIDTH/2], [NET_Y, NET_Y], [NET_HEIGHT, NET_HEIGHT], 'k', lw=2)
    ax.plot([-TABLE_WIDTH/2, TABLE_WIDTH/2], [NET_Y, NET_Y], [0, 0], 'k', lw=1)
    ax.plot([0, 0], [NET_Y, NET_Y], [0, NET_HEIGHT], 'k', lw=2)


# ========== 3面図 ==========
def plot_3views(trajectories):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

    # ===== 1. 3Dビュー =====
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

    # ===== 2. トップビュー (x–y) =====
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

    # ===== 3. サイドビュー (y–z) =====
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

    robo_pos = (0.5, 0, 0.2) # ロボットの位置
    target_my = None     # 1バウンド目（自分側）
    target_oppo = (-0.5, 2.0)  # 2バウンド目（相手側）
    v_list = np.arange(0.1, 5.0, 0.1)
    mode = "speed"
    target_speed = 3.0  # 目標速度 [m/s]

    start = time.time()
    best_v, best_theta, best_phi, score = find_best_serve_params(v_list, robo_pos, target_my, target_oppo, mode=mode, target=target_speed)
    end = time.time()
    print("探索時間:", end - start, "秒")
    print("最適速度:", best_v, "m/s")
    print("最適仰角:", np.rad2deg(best_theta), "度")
    print("最適横回転角:", np.rad2deg(best_phi), "度")
    print("評価スコア:", score)
    x, y, z, b = simulate_trajectory(best_v, best_theta, best_phi, robo_pos)
    plot_3views([{"x": x, "y": y, "z": z, "bounces": b}])

if __name__ == "__main__":
    main()