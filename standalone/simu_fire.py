import numpy as np
import matplotlib.pyplot as plt

g = 9.81

def simulate_ball(v0, angle_deg, h0=0, restitution=0.8, dt=0.001, t_max=10, target=None):
    """
    ボールの放物運動＋バウンドを計算して描画する関数
    v0: 初速度[m/s]
    angle_deg: 発射角度[度]
    h0: 発射高さ[m]
    restitution: 反発係数
    target: (x, y) 形式で目標地点を指定（省略可）
    """
    angle = np.deg2rad(angle_deg)
    vx = v0 * np.cos(angle)
    vy = v0 * np.sin(angle)
    
    x, y = [0], [h0]
    px, py = 0, h0
    t = 0
    
    while t < t_max:
        t += dt
        vy -= g * dt
        new_x = px + vx * dt
        new_y = py + vy * dt
        
        if new_y < 0:
            t_collision = -py / vy
            collision_x = px + vx * t_collision
            
            x.append(collision_x)
            y.append(0)
            
            remaining_t = dt - t_collision
            vy = -vy * restitution
            
            if abs(vy) < 0.5:
                break
            
            px = collision_x
            py = 0
            vy -= g * remaining_t
            new_x = px + vx * remaining_t
            new_y = py + vy * remaining_t
        
        x.append(new_x)
        y.append(max(0, new_y))
        px, py = new_x, new_y
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="Trajectory", color="blue", linewidth=2)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3)
    plt.title(f"Ball Trajectory (v0={v0} m/s, angle={angle_deg}°)")
    
    plt.scatter(0, h0, color="red", marker="o", s=100, label="Launch Point", zorder=5)
    plt.text(0, h0 + 1, "Launch", color="red", ha="center")
    
    if target is not None:
        tx, ty = target
        plt.scatter(tx, ty, color="green", marker="x", s=150, linewidths=3, label="Target Point", zorder=5)
        plt.text(tx, ty + 1, f"Target ({tx:.1f}, {ty:.1f})", color="green", ha="center")
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def calc_angle_for_target(v0, h0, target_x, target_y=0):
    """
    指定した速度で目標地点に到達するための発射角度を計算
    放物線の式: y = x*tan(θ) - g*x²/(2*v0²*cos²(θ)) + h0
    目標点で y = target_y となる θ を求める
    """
    dx = target_x
    dy = target_y - h0
    
    v0_sq = v0 ** 2
    g_dx_sq = g * dx ** 2
    
    a = g_dx_sq
    b = -2 * v0_sq * dx
    c = g_dx_sq + 2 * v0_sq * dy
    
    disc = b**2 - 4*a*c
    
    if disc < 0:
        print("指定条件では到達できません。速度を上げてください。")
        return None
    
    sqrt_disc = np.sqrt(disc)
    tan1 = (-b + sqrt_disc) / (2*a)
    tan2 = (-b - sqrt_disc) / (2*a)
    
    angles = []
    for tan_theta in [tan1, tan2]:
        angle_rad = np.arctan(tan_theta)
        angle_deg = np.rad2deg(angle_rad)
        if -90 < angle_deg < 90:
            angles.append(angle_deg)
    
    return angles if angles else None


height = 10.0
v0 = 50.0

#simulate_ball(v0=v0, angle_deg=30, h0=height)

target_x = 40
target_y = 0
angles = calc_angle_for_target(v0=v0, h0=height, target_x=target_x, target_y=target_y)

if angles:
    print("到達可能な発射角度:", angles)
    simulate_ball(v0=v0, angle_deg=angles[0], h0=height, target=(target_x, target_y))