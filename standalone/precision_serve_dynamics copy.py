import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, differential_evolution

class TableTennisSimulator:
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
        
        self.table_length = 2.74
        self.table_width = 1.525
        self.table_height = 0.76
        self.net_height = 0.1525
        
    def is_point_on_table(self, x, y, side='own'):
        if side == 'own':
            return -self.table_length/2 <= x <= 0 and -self.table_width/2 <= y <= self.table_width/2
        elif side == 'opponent':
            return 0 <= x <= self.table_length/2 and -self.table_width/2 <= y <= self.table_width/2
        return False
    
    def crossed_net(self, pos1, pos2):
        if pos1[0] <= 0 and pos2[0] > 0:
            t = -pos1[0] / (pos2[0] - pos1[0])
            crossing_height = pos1[2] + t * (pos2[2] - pos1[2])
            crossing_y = pos1[1] + t * (pos2[1] - pos1[1])
            
            if abs(crossing_y) <= self.table_width/2:
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
            
            if pos[2] < 0 or abs(pos[0]) > 5 or abs(pos[1]) > 3:
                break
        
        if len(bounces) == 0:
            serve_result['reason'] = 'No bounce on own side'
        elif len(bounces) == 1:
            if net_crossing is None:
                serve_result['reason'] = 'Did not cross the net (Net Miss)'
            elif not net_crossing:
                serve_result['reason'] = 'Hit the Net'
            else:
                serve_result['reason'] = 'No bounce on opponent side'
        else:
            if bounces[0]['side'] == 'own' and bounces[1]['side'] == 'opponent':
                if net_crossing:
                    serve_result['valid'] = True
                    serve_result['reason'] = 'Serve Successful'
                else:
                    serve_result['reason'] = 'Hit the Net'

        return np.array(trajectory), np.array(times), serve_result

def visualize_serve(trajectory, serve_result, simulator):
    fig = plt.figure(figsize=(15, 10))
    
    ax1 = fig.add_subplot(221, projection='3d')
    
    table_x = [-simulator.table_length/2, simulator.table_length/2, 
               simulator.table_length/2, -simulator.table_length/2, -simulator.table_length/2]
    table_y = [-simulator.table_width/2, -simulator.table_width/2, 
               simulator.table_width/2, simulator.table_width/2, -simulator.table_width/2]
    table_z = [simulator.table_height] * 5
    ax1.plot(table_x, table_y, table_z, 'k-', linewidth=2)
    
    net_y = [-simulator.table_width/2, simulator.table_width/2]
    net_z = [simulator.table_height, simulator.table_height]
    for y_coord in net_y:
        ax1.plot([0, 0], [y_coord, y_coord], 
                [simulator.table_height, simulator.table_height + simulator.net_height], 
                'gray', linewidth=2)
    ax1.plot([0, 0], net_y, 
            [simulator.table_height + simulator.net_height]*2, 
            'gray', linewidth=2)
    
    ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
             'b-', linewidth=2, label='Ball Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                c='g', s=100, marker='o', label='Start')
    
    if serve_result['own_bounce'] is not None:
        ob = serve_result['own_bounce']
        ax1.scatter(ob[0], ob[1], ob[2], c='orange', s=150, marker='v', label='Own Side Bounce')

    if serve_result['opponent_bounce'] is not None:
        opb = serve_result['opponent_bounce']
        ax1.scatter(opb[0], opb[1], opb[2], c='red', s=150, marker='v', label='Opponent Side Bounce')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.set_zlim(0, max(1.5, np.max(trajectory[:, 2]) * 1.2))
    
    ax2 = fig.add_subplot(222)
    ax2.axhline(y=simulator.table_height, color='brown', linewidth=3, label='Table')
    ax2.axvline(x=-simulator.table_length/2, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='gray', linewidth=2, label='Net Position')
    ax2.axvline(x=simulator.table_length/2, color='k', linestyle='--', alpha=0.3)
    ax2.plot([0, 0], [simulator.table_height, simulator.table_height + simulator.net_height],
             'gray', linewidth=3, label='Net')

    ax2.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2, label='Ball Trajectory')
    ax2.scatter(trajectory[0, 0], trajectory[0, 2], c='g', s=100, marker='o')
    
    if serve_result['own_bounce'] is not None:
        ob = serve_result['own_bounce']
        ax2.scatter(ob[0], ob[2], c='orange', s=150, marker='v')
    
    if serve_result['opponent_bounce'] is not None:
        opb = serve_result['opponent_bounce']
        ax2.scatter(opb[0], opb[2], c='red', s=150, marker='v')
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (X-Z)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(1.5, np.max(trajectory[:, 2]) * 1.2))
    
    ax3 = fig.add_subplot(223)
    table_rect = plt.Rectangle((-simulator.table_length/2, -simulator.table_width/2),
                                simulator.table_length, simulator.table_width,
                                fill=False, edgecolor='brown', linewidth=2)
    ax3.add_patch(table_rect)
    ax3.axvline(x=0, color='gray', linewidth=2, linestyle='--', label='Net Position')
    
    ax3.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Ball Trajectory')
    ax3.scatter(trajectory[0, 0], trajectory[0, 1], c='g', s=100, marker='o', label='Start')

    if serve_result['own_bounce'] is not None:
        ob = serve_result['own_bounce']
        ax3.scatter(ob[0], ob[1], c='orange', s=150, marker='v')
    
    if serve_result['opponent_bounce'] is not None:
        opb = serve_result['opponent_bounce']
        ax3.scatter(opb[0], opb[1], c='red', s=150, marker='v')
    
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Top View (X-Y)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    result_color = 'green' if serve_result['valid'] else 'red'
    result_text = f"Result:\n{serve_result['reason']}\n\n"
    
    if serve_result['own_bounce'] is not None:
        ob = serve_result['own_bounce']
        result_text += f"Own Bounce Position:\n  X={ob[0]:.3f}m, Y={ob[1]:.3f}m\n\n"
    
    if serve_result['net_crossing_height'] is not None:
        result_text += f"Net Crossing Height:\n  {serve_result['net_crossing_height']*100:.1f}cm (from table)\n\n"
    
    if serve_result['opponent_bounce'] is not None:
        opb = serve_result['opponent_bounce']
        result_text += f"Opponent Side Bounce Position:\n  X={opb[0]:.3f}m, Y={opb[1]:.3f}m"

    ax4.text(0.1, 0.5, result_text, fontsize=12, 
             verticalalignment='center', color=result_color,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

class ServeOptimizer:
    def __init__(self, simulator, initial_position):
        self.simulator = simulator
        self.initial_position = initial_position
        self.best_params = None
        self.best_result = None
        
    def objective_function(self, params):
        vx, vy, vz, omega_x, omega_y, omega_z = params
        
        velocity = [vx, vy, vz]
        angular_velocity = [omega_x, omega_y, omega_z]
        
        try:
            trajectory, times, serve_result = self.simulator.simulate_serve(
                self.initial_position,
                velocity,
                angular_velocity,
                max_duration=3.0
            )
            
            if serve_result['valid']:
                return 0.0
            
            penalty = 1000.0
            
            if serve_result['own_bounce'] is None:
                penalty += 500.0
            else:
                ob = serve_result['own_bounce']
                if abs(ob[1]) > self.simulator.table_width/2 * 0.8:
                    penalty += 100.0
            
            if serve_result['net_crossing_height'] is not None:
                if serve_result['net_crossing_height'] < 0:
                    penalty += abs(serve_result['net_crossing_height']) * 1000
                elif serve_result['net_crossing_height'] < 0.05:
                    penalty += (0.05 - serve_result['net_crossing_height']) * 500
            else:
                penalty += 800.0
            
            if serve_result['opponent_bounce'] is None and serve_result['own_bounce'] is not None:
                penalty += 300.0
            elif serve_result['opponent_bounce'] is not None:
                opb = serve_result['opponent_bounce']
                edge_distance_x = min(abs(opb[0] - 0), abs(opb[0] - self.simulator.table_length/2))
                edge_distance_y = self.simulator.table_width/2 - abs(opb[1])
                if edge_distance_x < 0.1 or edge_distance_y < 0.1:
                    penalty += 50.0
            
            return penalty
            
        except Exception as e:
            return 10000.0
    
    def optimize(self, initial_guess=None, method='differential_evolution'):
        bounds = [
            (1.5, 6.0),
            (-2.0, 2.0),
            (-3.0, 2.0),
            (-50.0, 50.0),
            (-200.0, 200.0),
            (-50.0, 50.0)
        ]
        
        print("最適化を開始します...")
        print(f"方法: {method}")
        
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                bounds,
                maxiter=100,
                popsize=15,
                tol=0.01,
                seed=42,
                workers=1,
                updating='deferred',
                callback=self._callback
            )
        else:
            if initial_guess is None:
                initial_guess = [3.5, 0.0, -1.5, 0.0, -100.0, 0.0]
            
            result = minimize(
                self.objective_function,
                initial_guess,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.1}
            )
        
        self.best_params = result.x
        
        velocity = [result.x[0], result.x[1], result.x[2]]
        angular_velocity = [result.x[3], result.x[4], result.x[5]]
        
        trajectory, times, serve_result = self.simulator.simulate_serve(
            self.initial_position,
            velocity,
            angular_velocity
        )
        
        self.best_result = serve_result
        
        print("\n" + "="*60)
        print("最適化完了！")
        print("="*60)
        print(f"\n最適パラメータ:")
        print(f"  初速度: vx={velocity[0]:.3f}, vy={velocity[1]:.3f}, vz={velocity[2]:.3f} m/s")
        print(f"  角速度: ωx={angular_velocity[0]:.1f}, ωy={angular_velocity[1]:.1f}, ωz={angular_velocity[2]:.1f} rad/s")
        print(f"\n判定: {serve_result['reason']}")
        print(f"成功: {'✓' if serve_result['valid'] else '✗'}")
        
        if serve_result['net_crossing_height'] is not None:
            print(f"ネット通過高さ: {serve_result['net_crossing_height']*100:.1f}cm")
        
        return trajectory, times, serve_result
    
    def _callback(self, xk, convergence):
        if hasattr(self, '_iteration'):
            self._iteration += 1
        else:
            self._iteration = 1
        
        if self._iteration % 10 == 0:
            score = self.objective_function(xk)
            print(f"  イテレーション {self._iteration}: スコア = {score:.2f}")
        return False

if __name__ == "__main__":
    simulator = TableTennisSimulator()
    
    print("="*60)
    print("卓球サーブ最適化システム")
    print("="*60)
    
    mode = input("\n実行モード選択:\n  1: 手動パラメータでシミュレーション\n  2: 失敗サーブを最適化\n  3: 初期位置から最適なサーブを探索\n選択 (1/2/3): ").strip()
    
    if mode == '1':
        initial_position = [-0.5, 0.0, 1.0]
        initial_velocity = [3.5, 0.0, -1.5]
        angular_velocity = [0.0, -100.0, 0.0]
        
        print("\n手動パラメータでシミュレーション...")
        print(f"初期位置: {initial_position}")
        print(f"初速度: {initial_velocity} m/s")
        print(f"角速度: {angular_velocity} rad/s")
        
        trajectory, times, serve_result = simulator.simulate_serve(
            initial_position,
            initial_velocity,
            angular_velocity
        )
        
        print(f"\n{'='*50}")
        print(f"判定結果: {serve_result['reason']}")
        print(f"サーブ成功: {'✓' if serve_result['valid'] else '✗'}")
        print(f"{'='*50}")
        
        visualize_serve(trajectory, serve_result, simulator)
    
    elif mode == '2':
        print("\n失敗するサーブパラメータを入力してください")
        initial_position = list(map(float, input("初期位置 [x y z] (例: -0.5 0 1.0): ").split()))
        initial_velocity = list(map(float, input("初速度 [vx vy vz] (例: 5 0 0): ").split()))
        angular_velocity = list(map(float, input("角速度 [ωx ωy ωz] (例: 0 -50 0): ").split()))
        
        print("\n元のパラメータでシミュレーション...")
        trajectory_orig, times_orig, serve_result_orig = simulator.simulate_serve(
            initial_position,
            initial_velocity,
            angular_velocity
        )
        
        print(f"判定: {serve_result_orig['reason']}")
        
        if not serve_result_orig['valid']:
            print("\nサーブが失敗しました。最適化を開始します...")
            
            optimizer = ServeOptimizer(simulator, initial_position)
            initial_guess = initial_velocity + angular_velocity
            trajectory_opt, times_opt, serve_result_opt = optimizer.optimize(
                initial_guess=initial_guess,
                method='Nelder-Mead'
            )
            
            if serve_result_opt['valid']:
                print("\n成功するパラメータが見つかりました！")
                visualize_serve(trajectory_opt, serve_result_opt, simulator)
            else:
                print("\n警告: 最適化後もサーブが成功しませんでした。")
                print("より広範囲な探索を試みます...")
                
                trajectory_opt, times_opt, serve_result_opt = optimizer.optimize(
                    method='differential_evolution'
                )
                visualize_serve(trajectory_opt, serve_result_opt, simulator)
        else:
            print("\nこのサーブは既に成功しています！")
            visualize_serve(trajectory_orig, serve_result_orig, simulator)
    
    elif mode == '3':
        print("\n初期位置を入力してください")
        initial_position = list(map(float, input("初期位置 [x y z] (例: -0.5 0 1.0): ").split()))
        
        optimizer = ServeOptimizer(simulator, initial_position)
        trajectory, times, serve_result = optimizer.optimize(method='differential_evolution')
        
        if serve_result['valid']:
            print("\n✓ 成功するサーブパラメータが見つかりました！")
        else:
            print("\n警告: 最適化後もサーブが成功しませんでした。")
            print("パラメータ範囲や制約を見直す必要があるかもしれません。")
        
        visualize_serve(trajectory, serve_result, simulator)
    
    else:
        print("無効な選択です。")