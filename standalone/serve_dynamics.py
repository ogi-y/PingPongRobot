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

# ボール物性（標準値：卓球ボール）
R_BALL = 0.02           # 半径 [m]（40mm球）
A_CROSS = np.pi * R_BALL**2  # 正面投影面積 [m^2]
M_BALL = 0.0027         # 質量 [kg]（約2.7 g）
RHO_AIR = 1.225         # 空気密度 [kg/m^3]

# ========== 軌道シミュレーション（ドラッグ＋マグヌス） ==========
def simulate_trajectory(v0, theta, phi, robo_pos,
                        omega_local=(0.0, 0.0, 0.0),
                        C_d=0.5, C_l=0.2,
                        m=M_BALL,
                        restitution=0.9,
                        spin_restitution=0.9,
                        tangential_friction=0.9,
                        max_bounce=3,
                        y_limit=TABLE_LENGTH*2,
                        dt=0.001,
                        max_time=10.0):
    """
    空気抵抗とマグヌス効果を含む数値シミュレーション（ニュートン運動方程式に基づく）
    - v0: 初速の大きさ [m/s]
    - theta: 仰角 [rad]
    - phi: 水平角（右が正） [rad]
    - robo_pos: 初期位置 (x, y, z)
    - omega: 回転ベクトル (wx, wy, wz) [rad/s] （右手系）
    - C_d: 抗力係数（球で ~0.4-0.6 の範囲）
    - C_l: 揚力（マグヌス）係数のスケーリング（経験値で 0.0-0.4 程度）
    - m: ボール質量 [kg]
    - restitution: 法線方向の反発係数
    - spin_restitution: バウンド後の回転保持率（0-1）
    - tangential_friction: バウンドでの接線速度の減衰（0-1）
    - dt: タイムステップ [s]
    - max_time: 最大シミュレーション時間 [s]
    """

    def local_to_global(omega_local):
        v_dir = np.array([
            np.cos(theta) * np.sin(phi),
            np.cos(theta) * np.cos(phi),
            np.sin(theta)
        ])
        v_dir /= np.linalg.norm(v_dir)

        # ローカル座標→ワールド座標変換
        world_up = np.array([0, 0, 1])
        right = np.cross(world_up, v_dir)
        right /= np.linalg.norm(right)
        up = np.cross(v_dir, right)

        omega_local = np.array(omega_local)
        omega = (
            v_dir * omega_local[0] +
            right * omega_local[1] +
            up * omega_local[2]
        )

        return omega

    omega = local_to_global(omega_local)
    # 初期速度ベクトル（球面座標 -> カート座標）
    vx = v0 * np.cos(theta) * np.sin(phi)
    vy = v0 * np.cos(theta) * np.cos(phi)
    vz = v0 * np.sin(theta)

    vel = np.array([vx, vy, vz], dtype=float)
    pos = np.array([robo_pos[0], robo_pos[1], robo_pos[2]], dtype=float)
    omega = np.array(omega, dtype=float)  # 回転ベクトル

    xs, ys, zs = [pos[0]], [pos[1]], [pos[2]]
    bounces = []
    t = 0.0

    # 安全上限ステップ数
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

        # 重力
        F_grav = np.array([0.0, 0.0, -m * g])

        # 合力 -> 加速度
        F_total = F_drag + F_magnus + F_grav
        acc = F_total / m

        # 時刻進める（単純な陽的オイラー；必要なら RK4 に拡張）
        vel = vel + acc * dt
        pos = pos + vel * dt
        t += dt

        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

        # 台（z=0）との衝突判定
        if pos[2] <= 0 and vel[2] < 0:
            # 衝突位置を補間してもう少し正確に当てる（簡易）
            # 補間係数 alpha: pos_old + alpha * vel_old*dt => z==0 を解く
            z_prev = zs[-2]
            v_prev_z = (zs[-1] - zs[-2]) / dt  # 近似
            if abs(v_prev_z) > 1e-9:
                alpha = z_prev / (z_prev - pos[2])  # 0<=alpha<=1 なら補間可
                # 位置補正（より正確にバウンド位置を取る）
                pos[0] = xs[-2] + (pos[0] - xs[-2]) * alpha
                pos[1] = ys[-2] + (pos[1] - ys[-2]) * alpha
                pos[2] = 0.0
            else:
                pos[2] = 0.0

            # 法線方向（z）の反発
            vel_normal = np.array([0.0, 0.0, vel[2]])
            vel_tangent = vel - vel_normal

            # 反発係数（法線）
            vel_normal[2] = -vel_normal[2] * restitution

            # 接線方向は摩擦で減衰させる（ラフにモデル化）
            vel_tangent = vel_tangent * tangential_friction

            # 角速度の変化（簡易）：バウンドで一部減衰
            omega = omega * spin_restitution

            # 合成して新しい速度
            vel = vel_tangent + vel_normal

            bounces.append((pos[0], pos[1]))
            # ここでバウンド回数チェック
            if len(bounces) >= max_bounce:
                break

            # 小さな上向き速度が得られた場合は継続（次の飛行）
        # y方向の距離制限で停止（前方判定）
        if pos[1] >= y_limit:
            break

        # ありえない高さで停止（安全）
        if pos[2] > 5.0:
            break

        # 時間上限
        if t >= max_time:
            break

    return np.array(xs), np.array(ys), np.array(zs), bounces

# evaluate_serve 等は引数の互換性を保つためにほぼ変えずに利用できるようにします。
def evaluate_serve(v, theta, phi, robo_pos, target_my = None, target_oppo = None, mode = "speed", target = None, mode_alpha=1.0,
                   **sim_kwargs):
    """
    simulate_trajectory に渡す追加引数は sim_kwargs へ。
    """
    x, y, z, bounces = simulate_trajectory(v, theta, phi, robo_pos, **sim_kwargs)
    BAD_SCORE = -10000.0
    SAFE_HEIGHT = 0.05  # ネット上を通過するための余裕

    if len(bounces) < 2:
        return BAD_SCORE
    for bx, by in bounces[:2]:
        if bx < -TABLE_WIDTH/2 or bx > TABLE_WIDTH/2:
            return BAD_SCORE
        if by < 0 or by > TABLE_LENGTH:
            return BAD_SCORE

    # ネット位置に最も近い y のインデックス
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
                              theta_min_deg=-45, theta_max_deg=45, steps=45, mode=None, target=None, **sim_kwargs):
    best_theta = None
    best_score = -1e9

    thetas = np.linspace(np.deg2rad(theta_min_deg), np.deg2rad(theta_max_deg), steps)

    for theta in thetas:
        score = evaluate_serve(v, theta, phi, robo_pos, target_my, target_oppo, mode=mode, target=target, **sim_kwargs)
        if score > best_score:
            best_score = score
            best_theta = theta

    return best_theta, best_score

def find_best_serve_params(v_list, robo_pos, target_my, target_oppo, mode="speed", target=10.0, **sim_kwargs):
    best_v = None
    best_theta = None
    best_phi = None
    best_score = -1e9

    # xy平面上でphiを求める（ターゲットが与えられたとき）
    if target_oppo is not None:
        phi = np.arctan2(target_oppo[0] - robo_pos[0], target_oppo[1] - robo_pos[1])
    elif target_my is not None:
        phi = np.arctan2(target_my[0] - robo_pos[0], target_my[1] - robo_pos[1])
    else:
        phi = 0.0

    for v in v_list:
        theta, score = find_best_theta(v, phi, robo_pos, target_my, target_oppo,
                                        mode=mode, target=target, **sim_kwargs)
        if score > best_score:
            best_score = score
            best_theta = theta
            best_phi = phi
            best_v = v

    return best_v, best_theta, best_phi, best_score

# ========== 描画関係は変えていません（元コードのまま利用） ==========
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
    robo_pos = (0.2, 0, 0.27) # ロボットの位置
    target_my = None     # 1バウンド目（自分側）
    target_oppo = (0.5, 2.0)  # 2バウンド目（相手側）
    v_list = np.arange(0.0, 6.0, 1.0)
    mode = "speed"
    target_speed = 3.0  # 目標速度 [m/s]

    # 例: 少しトップスピン（x軸周りの負の回転）、少し側回転を混ぜた回転ベクトル
    # omega = (wx, wy, wz) [rad/s]。w ~ 50-200 rad/s は卓球ボールであり得る範囲（試験的に）
    sim_kwargs = {
        "omega_local": (0.0, 80.0, -200.0),  # (wx, wy, wz) - 調整してみてください
        "C_d": 0.5,
        "C_l": 0.18,
        "m": M_BALL,
        "restitution": 0.9,
        "spin_restitution": 0.85,
        "tangential_friction": 0.9,
        "dt": 0.005,
        "max_time": 2.0
    }

    start = time.time()
    best_v, best_theta, best_phi, score = find_best_serve_params(v_list, robo_pos, target_my, target_oppo, mode=mode, target=target_speed, **sim_kwargs)
    end = time.time()
    print("探索時間:", end - start, "秒")
    print("最適速度:", best_v, "m/s")
    print("最適仰角:", np.rad2deg(best_theta), "度")
    print("最適横回転角:", np.rad2deg(best_phi), "度")
    print("評価スコア:", score)
    x, y, z, b = simulate_trajectory(best_v, best_theta, best_phi, robo_pos, **sim_kwargs)
    plot_3views([{"x": x, "y": y, "z": z, "bounces": b}])

if __name__ == "__main__":
    main()
