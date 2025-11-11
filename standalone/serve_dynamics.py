import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from scipy.optimize import minimize

g = 9.80665

TABLE_LENGTH = 2.74
TABLE_WIDTH = 1.525
NET_Y = TABLE_LENGTH / 2
NET_HEIGHT = 0.1525

def find_bounce_time(z0, vz, restitution_power=0):
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
    vx0 = v * np.cos(theta) * np.sin(phi)
    vy0 = v * np.cos(theta) * np.cos(phi)
    vz0 = v * np.sin(theta)
    
    x0, y0, z0 = robo_pos
    
    xs, ys, zs = [], [], []
    bounces = []
    
    vx, vy, vz = vx0, vy0, vz0
    x, y, z = x0, y0, z0
    
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


class PreciseTableTennisSimulator:
    def __init__(self, dt=0.0001):
        self.mass = 0.0027
        self.radius = 0.020
        self.dt = dt
        
        self.g = 9.81
        self.air_density = 1.225
        self.drag_coefficient = 0.5
        self.magnus_coefficient = 0.6
        
        self.restitution = 0.89
        self.friction_coefficient = 0.4
        
        self.I = (2/5) * self.mass * self.radius**2
        
        self.table_length = TABLE_LENGTH
        self.table_width = TABLE_WIDTH
        self.table_height = 0.0
        self.net_height = NET_HEIGHT
        self.net_y = NET_Y
        
    def is_point_on_table(self, x, y, side='own'):
        if side == 'own':
            return 0 <= y <= self.net_y and -self.table_width/2 <= x <= self.table_width/2
        elif side == 'opponent':
            return self.net_y <= y <= self.table_length and -self.table_width/2 <= x <= self.table_width/2
        return False
    
    def crossed_net(self, pos1, pos2):
        if pos1[1] <= self.net_y and pos2[1] > self.net_y:
            t = (self.net_y - pos1[1]) / (pos2[1] - pos1[1])
            crossing_height = pos1[2] + t * (pos2[2] - pos1[2])
            crossing_x = pos1[0] + t * (pos2[0] - pos1[0])
            
            if abs(crossing_x) <= self.table_width/2:
                return crossing_height > self.table_height + self.net_height
        return None
    
    def simulate_serve(self, position, velocity, angular_velocity, max_duration=3.0):
        pos = np.array(position, dtype=float)
        vel = np.array(velocity, dtype=float)
        omega = np.array(angular_velocity, dtype=float)
        
        trajectory = [pos.copy()]
        times = [0]
        
        bounces = []
        net_crossing = None
        serve_result = {
            'valid': False,
            'reason': '',
            'own_bounce': None,
            'opponent_bounce': None,
            'net_crossing_height': None
        }
        
        t = 0
        prev_pos = pos.copy()
        
        while t < max_duration:
            speed = np.linalg.norm(vel)
            
            drag_force = -0.5 * self.air_density * self.drag_coefficient * \
                         np.pi * self.radius**2 * speed * vel
            
            if speed > 0.001:
                omega_magnitude = np.linalg.norm(omega)
                if omega_magnitude > 0.001:
                    spin_axis = omega / omega_magnitude
                    magnus_direction = np.cross(spin_axis, vel / speed)
                    magnus_force = self.magnus_coefficient * np.pi * self.radius**3 * \
                                  self.air_density * omega_magnitude * speed * magnus_direction
                else:
                    magnus_force = np.zeros(3)
            else:
                magnus_force = np.zeros(3)
            
            gravity_force = np.array([0, 0, -self.mass * self.g])
            
            total_force = drag_force + magnus_force + gravity_force
            acceleration = total_force / self.mass
            
            vel += acceleration * self.dt
            pos += vel * self.dt
            
            net_cross = self.crossed_net(prev_pos, pos)
            if net_cross is not None and net_crossing is None:
                net_crossing = net_cross
                serve_result['net_crossing_height'] = net_cross - self.table_height
            
            if pos[2] <= self.table_height + self.radius:
                pos[2] = self.table_height + self.radius
                
                if self.is_point_on_table(pos[0], pos[1], 'own'):
                    if len(bounces) == 0:
                        bounces.append({
                            'position': pos.copy(),
                            'time': t,
                            'side': 'own'
                        })
                        serve_result['own_bounce'] = pos.copy()
                
                elif self.is_point_on_table(pos[0], pos[1], 'opponent'):
                    if len(bounces) == 1 and bounces[0]['side'] == 'own':
                        bounces.append({
                            'position': pos.copy(),
                            'time': t,
                            'side': 'opponent'
                        })
                        serve_result['opponent_bounce'] = pos.copy()
                
                normal = np.array([0, 0, 1])
                vel_normal = np.dot(vel, normal) * normal
                vel_tangent = vel - vel_normal
                
                vel_normal = -self.restitution * vel_normal
                
                contact_point_velocity = vel_tangent + np.cross(omega, self.radius * normal)
                relative_velocity_magnitude = np.linalg.norm(contact_point_velocity)
                
                if relative_velocity_magnitude > 0.001:
                    friction_direction = -contact_point_velocity / relative_velocity_magnitude
                    friction_force = self.friction_coefficient * self.mass * self.g * friction_direction
                    
                    vel_tangent += friction_force / self.mass * self.dt
                    
                    torque = np.cross(self.radius * normal, friction_force)
                    angular_acceleration = torque / self.I
                    omega += angular_acceleration * self.dt
                
                vel = vel_normal + vel_tangent
                omega *= 0.95
            
            prev_pos = pos.copy()
            t += self.dt
            
            if len(trajectory) < 1 or t - times[-1] >= 0.002:
                trajectory.append(pos.copy())
                times.append(t)
            
            if len(bounces) >= 2:
                break
            
            if pos[2] < 0 or abs(pos[0]) > 5 or abs(pos[1]) > 5:
                break
        
        if len(bounces) == 0:
            serve_result['reason'] = '自陣でバウンドしていません'
        elif len(bounces) == 1:
            if net_crossing is None:
                serve_result['reason'] = 'ネットを越えていません'
            elif not net_crossing:
                serve_result['reason'] = 'ネットに当たりました'
            else:
                serve_result['reason'] = '相手陣地でバウンドしていません'
        else:
            if bounces[0]['side'] == 'own' and bounces[1]['side'] == 'opponent':
                if net_crossing:
                    serve_result['valid'] = True
                    serve_result['reason'] = 'サーブ成功'
                else:
                    serve_result['reason'] = 'ネットに当たりました'
        
        return np.array(trajectory), np.array(times), serve_result


class HybridServeOptimizer:
    def __init__(self, robo_pos):
        self.robo_pos = robo_pos
        self.precise_simulator = PreciseTableTennisSimulator()
        
    def velocity_to_params(self, vx, vy, vz):
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        theta = np.arcsin(vz / v) if v > 0 else 0
        phi = np.arctan2(vx, vy) if vy != 0 or vx != 0 else 0
        return v, theta, phi
    
    def params_to_velocity(self, v, theta, phi):
        vx = v * np.cos(theta) * np.sin(phi)
        vy = v * np.cos(theta) * np.cos(phi)
        vz = v * np.sin(theta)
        return vx, vy, vz
    
    def verify_with_precise_simulation(self, v, theta, phi, omega_x=0, omega_y=-100, omega_z=0):
        vx, vy, vz = self.params_to_velocity(v, theta, phi)
        velocity = [vx, vy, vz]
        angular_velocity = [omega_x, omega_y, omega_z]
        
        trajectory, times, serve_result = self.precise_simulator.simulate_serve(
            self.robo_pos,
            velocity,
            angular_velocity
        )
        
        return trajectory, times, serve_result
    
    def objective_function_precise(self, params):
        vx, vy, vz = params[:3]
        omega_x, omega_y, omega_z = params[3:] if len(params) > 3 else [0, -100, 0]
        
        velocity = [vx, vy, vz]
        angular_velocity = [omega_x, omega_y, omega_z]
        
        try:
            trajectory, times, serve_result = self.precise_simulator.simulate_serve(
                self.robo_pos,
                velocity,
                angular_velocity,
                max_duration=3.0
            )
            
            if serve_result['valid']:
                return 0.0
            
            penalty = 1000.0
            
            if serve_result['own_bounce'] is None:
                penalty += 500.0
            
            if serve_result['net_crossing_height'] is not None:
                if serve_result['net_crossing_height'] < 0:
                    penalty += abs(serve_result['net_crossing_height']) * 1000
                elif serve_result['net_crossing_height'] < 0.05:
                    penalty += (0.05 - serve_result['net_crossing_height']) * 500
            else:
                penalty += 800.0
            
            if serve_result['opponent_bounce'] is None and serve_result['own_bounce'] is not None:
                penalty += 300.0
            
            return penalty
            
        except Exception as e:
            return 10000.0
    
    def fine_tune_params(self, v_init, theta_init, phi_init, omega_init=[0, -100, 0]):
        vx_init, vy_init, vz_init = self.params_to_velocity(v_init, theta_init, phi_init)
        
        initial_guess = [vx_init, vy_init, vz_init] + omega_init
        
        bounds = [
            (0.5, 8.0),
            (0.5, 8.0),
            (-5.0, 5.0),
            (-100.0, 100.0),
            (-300.0, 300.0),
            (-100.0, 100.0)
        ]
        
        print("\n精密シミュレーションによる微調整を開始...")
        
        result = minimize(
            self.objective_function_precise,
            initial_guess,
            method='Nelder-Mead',
            bounds=bounds,
            options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.1}
        )
        
        optimized_params = result.x
        vx, vy, vz = optimized_params[:3]
        omega_x, omega_y, omega_z = optimized_params[3:]
        
        v_opt, theta_opt, phi_opt = self.velocity_to_params(vx, vy, vz)
        
        trajectory, times, serve_result = self.verify_with_precise_simulation(
            v_opt, theta_opt, phi_opt, omega_x, omega_y, omega_z
        )
        
        return v_opt, theta_opt, phi_opt, [omega_x, omega_y, omega_z], trajectory, times, serve_result


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

def plot_comparison(analytical_traj, precise_traj, serve_result):
    fig = plt.figure(figsize=(18, 6))
    
    ax1 = fig.add_subplot(131, projection='3d')
    draw_table_3d(ax1)
    ax1.plot(analytical_traj["x"], analytical_traj["y"], analytical_traj["z"], 
             'b-', linewidth=2, label='Analytical Model', alpha=0.7)
    if precise_traj is not None:
        ax1.plot(precise_traj[:, 0], precise_traj[:, 1], precise_traj[:, 2], 
                 'r-', linewidth=2, label='Precise Model')
    for j, (xb, yb) in enumerate(analytical_traj["bounces"][:2]):
        ax1.scatter(xb, yb, 0, color='orange' if j == 0 else 'green', s=100, marker='v')
    if serve_result and serve_result['own_bounce'] is not None:
        ob = serve_result['own_bounce']
        ax1.scatter(ob[0], ob[1], ob[2], c='red', s=150, marker='*', label='Precise Bounce')
    if serve_result and serve_result['opponent_bounce'] is not None:
        opb = serve_result['opponent_bounce']
        ax1.scatter(opb[0], opb[1], opb[2], c='purple', s=150, marker='*')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Comparison')
    ax1.legend()
    ax1.set_xlim(-TABLE_WIDTH/2, TABLE_WIDTH/2)
    ax1.set_ylim(0, TABLE_LENGTH)
    ax1.set_zlim(0, 1.0)
    
    ax2 = fig.add_subplot(132)
    ax2.axhline(y=NET_Y, color='gray', linewidth=2, linestyle='--', label='Net')
    ax2.plot(analytical_traj["x"], analytical_traj["y"], 'b-', linewidth=2, label='Analytical', alpha=0.7)
    if precise_traj is not None:
        ax2.plot(precise_traj[:, 0], precise_traj[:, 1], 'r-', linewidth=2, label='Precise')
    for j, (xb, yb) in enumerate(analytical_traj["bounces"][:2]):
        ax2.scatter(xb, yb, color='orange' if j == 0 else 'green', s=100, marker='v')
    ax2.add_patch(plt.Rectangle((-TABLE_WIDTH/2, 0), TABLE_WIDTH, TABLE_LENGTH,
                                fill=False, edgecolor='brown', linewidth=2))
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    ax3 = fig.add_subplot(133)
    ax3.axvline(x=NET_Y, color='gray', linewidth=2, linestyle='--', label='Net')
    ax3.axhline(y=NET_HEIGHT, color='gray', linewidth=1, linestyle=':')
    ax3.plot(analytical_traj["y"], analytical_traj["z"], 'b-', linewidth=2, label='Analytical', alpha=0.7)
    if precise_traj is not None:
        ax3.plot(precise_traj[:, 1], precise_traj[:, 2], 'r-', linewidth=2, label='Precise')
    for j, (xb, yb) in enumerate(analytical_traj["bounces"][:2]):
        ax3.scatter(yb, 0, color='orange' if j == 0 else 'green', s=100, marker='v')
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.show()

def main():
    robo_pos = (0.0, 0.0, 0.27)
    target_my = None
    target_oppo = (0.5, 2.5)
    v_list = np.arange(2.0, 5.0, 0.2)
    omega = [0, -100, 300]
    
    print("="*70)
    print("2段階卓球サーブ最適化システム")
    print("="*70)
    
    print("\n【STEP 1】簡易モデルで高速探索...")
    start = time.time()
    best_v, best_theta, best_phi, score = find_best_serve_params(
        v_list, robo_pos, target_my, target_oppo, mode="speed", target=0
    )
    end = time.time()
    
    print(f"探索時間: {end - start:.3f} 秒")
    print(f"最適速度: {best_v:.3f} m/s")
    print(f"最適仰角: {np.rad2deg(best_theta):.2f} 度")
    print(f"最適方位角: {np.rad2deg(best_phi):.2f} 度")
    print(f"評価スコア: {score:.2f}")
    
    x, y, z, bounces = simulate_trajectory_analytical(best_v, best_theta, best_phi, robo_pos)
    analytical_traj = {"x": x, "y": y, "z": z, "bounces": bounces}
    
    print("\n【STEP 2】精密モデルで検証...")
    optimizer = HybridServeOptimizer(robo_pos)
    
    trajectory_init, times_init, serve_result_init = optimizer.verify_with_precise_simulation(
        best_v, best_theta, best_phi, omega_x=omega[0], omega_y=omega[1], omega_z=omega[2]
    )
    
    print(f"初期パラメータでの判定: {serve_result_init['reason']}")
    print(f"成功: {'✓' if serve_result_init['valid'] else '✗'}")
    
    if not serve_result_init['valid']:
        print("\n【STEP 3】精密シミュレーションで微調整...")
        start_tune = time.time()
        v_opt, theta_opt, phi_opt, omega_opt, trajectory_opt, times_opt, serve_result_opt = \
            optimizer.fine_tune_params(best_v, best_theta, best_phi, omega_init=[0, -100, 0])
        end_tune = time.time()
        
        print(f"\n微調整時間: {end_tune - start_tune:.3f} 秒")
        print(f"調整後速度: {v_opt:.3f} m/s (変化: {v_opt-best_v:+.3f})")
        print(f"調整後仰角: {np.rad2deg(theta_opt):.2f} 度 (変化: {np.rad2deg(theta_opt-best_theta):+.2f})")
        print(f"調整後方位角: {np.rad2deg(phi_opt):.2f} 度 (変化: {np.rad2deg(phi_opt-best_phi):+.2f})")
        print(f"最適角速度: ωx={omega_opt[0]:.1f}, ωy={omega_opt[1]:.1f}, ωz={omega_opt[2]:.1f} rad/s")
        print(f"\n最終判定: {serve_result_opt['reason']}")
        print(f"成功: {'✓' if serve_result_opt['valid'] else '✗'}")
        
        if serve_result_opt['net_crossing_height'] is not None:
            print(f"ネット通過高さ: {serve_result_opt['net_crossing_height']*100:.1f} cm")
        
        plot_comparison(analytical_traj, trajectory_opt, serve_result_opt)
    else:
        print("\n✓ 簡易モデルのパラメータで既に成功しています！")
        plot_comparison(analytical_traj, trajectory_init, serve_result_init)

if __name__ == "__main__":
    main()