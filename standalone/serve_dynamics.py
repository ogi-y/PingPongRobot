import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from scipy.optimize import minimize
from enum import Enum

@dataclass
class Point3D:
    x: float
    y: float
    z: float

@dataclass
class Solution:
    v0: float
    azimuth: float
    elevation: float
    trajectory: List[Point3D]
    
    @property
    def total_angle(self):
        return np.sqrt(self.azimuth**2 + self.elevation**2)

class OptimizationStrategy(Enum):
    """最適化戦略の種類"""
    MIN_ANGLE = "minimum_total_angle"
    MIN_ELEVATION = "minimum_elevation"
    MAX_VELOCITY = "maximum_velocity"
    MIN_VELOCITY = "minimum_velocity"
    TARGET_ANGLE = "target_angle"
    TARGET_VELOCITY = "target_velocity"
    CUSTOM = "custom"

class BallisticCalculator3D:
    def __init__(self, g: float = 9.81):
        self.g = g
    
    def solve_angles(self, v0: float, target: Point3D, start_z: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
        horizontal_dist = np.sqrt(target.x**2 + target.y**2)
        dz = target.z - start_z
        
        azimuth = np.arctan2(target.y, target.x)
        
        a = self.g * horizontal_dist * horizontal_dist / (2 * v0 * v0)
        b = -horizontal_dist
        c = a + dz
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        tan_elev1 = (-b + np.sqrt(discriminant)) / (2 * a)
        tan_elev2 = (-b - np.sqrt(discriminant)) / (2 * a)
        
        elev1 = np.arctan(tan_elev1)
        elev2 = np.arctan(tan_elev2)
        
        high = max(elev1, elev2)
        low = min(elev1, elev2)
        
        return (azimuth, high, azimuth, low)
    
    def calculate_trajectory(self, v0: float, azimuth: float, elevation: float, 
                            start_z: float, max_dist: float, steps: int = 100) -> List[Point3D]:
        vx = v0 * np.cos(elevation) * np.cos(azimuth)
        vy = v0 * np.cos(elevation) * np.sin(azimuth)
        vz = v0 * np.sin(elevation)
        
        v_horizontal = np.sqrt(vx**2 + vy**2)
        
        if v_horizontal > 0:
            time_of_flight = (vz + np.sqrt(vz * vz + 2 * self.g * start_z)) / self.g
        else:
            time_of_flight = 0
        
        trajectory = []
        for i in range(steps + 1):
            t = (time_of_flight * i) / steps
            x = vx * t
            y = vy * t
            z = start_z + vz * t - 0.5 * self.g * t * t
            
            dist = np.sqrt(x**2 + y**2)
            if dist <= max_dist * 1.1 and z >= 0:
                trajectory.append(Point3D(x=x, y=y, z=z))
        
        return trajectory
    
    def find_solutions(self, velocities: List[float], target: Point3D, 
                      start_z: float = 0.0, allow_neg_elev: bool = True) -> List[Solution]:
        solutions = []
        max_dist = np.sqrt(target.x**2 + target.y**2)
        
        for v0 in velocities:
            angles = self.solve_angles(v0, target, start_z)
            
            if angles is None:
                continue
            
            azimuth_high, elev_high, azimuth_low, elev_low = angles
            
            elev_high_deg = elev_high * 180 / np.pi
            elev_low_deg = elev_low * 180 / np.pi
            
            min_elev = -90.0 if allow_neg_elev else 0.0
            max_elev = 90.0

            if min_elev <= elev_high_deg <= max_elev:
                trajectory = self.calculate_trajectory(v0, azimuth_high, elev_high, start_z, max_dist)
                solutions.append(Solution(
                    v0=v0,
                    azimuth=azimuth_high * 180 / np.pi,
                    elevation=elev_high_deg,
                    trajectory=trajectory
                ))

            if min_elev <= elev_low_deg <= max_elev and abs(elev_high_deg - elev_low_deg) > 0.1:
                trajectory = self.calculate_trajectory(v0, azimuth_low, elev_low, start_z, max_dist)
                solutions.append(Solution(
                    v0=v0,
                    azimuth=azimuth_low * 180 / np.pi,
                    elevation=elev_low_deg,
                    trajectory=trajectory
                ))
        
        return solutions
    
    def select_best_solution(self, solutions: List[Solution], 
                            strategy: OptimizationStrategy = OptimizationStrategy.MIN_ANGLE,
                            target_elevation: Optional[float] = None,
                            target_velocity: Optional[float] = None,
                            custom_scorer: Optional[Callable[[Solution], float]] = None) -> Optional[Solution]:
        """
        複数の解から最適な解を選択
        
        Args:
            solutions: 候補解のリスト
            strategy: 選択戦略
            target_elevation: 目標仰角（度）、TARGET_ANGLE戦略で使用
            target_velocity: 目標速度（m/s）、TARGET_VELOCITY戦略で使用
            custom_scorer: カスタム評価関数（値が小さいほど良い）
        
        Returns:
            最適な解、または None
        """
        if not solutions:
            return None
        
        if strategy == OptimizationStrategy.MIN_ANGLE:
            return min(solutions, key=lambda s: s.total_angle)
        
        elif strategy == OptimizationStrategy.MIN_ELEVATION:
            return min(solutions, key=lambda s: s.elevation)
        
        elif strategy == OptimizationStrategy.MAX_VELOCITY:
            return max(solutions, key=lambda s: s.v0)
        
        elif strategy == OptimizationStrategy.MIN_VELOCITY:
            return min(solutions, key=lambda s: s.v0)
        
        elif strategy == OptimizationStrategy.TARGET_ANGLE:
            if target_elevation is None:
                raise ValueError("target_elevation must be provided for TARGET_ANGLE strategy")
            return min(solutions, key=lambda s: abs(s.elevation - target_elevation))
        
        elif strategy == OptimizationStrategy.TARGET_VELOCITY:
            if target_velocity is None:
                raise ValueError("target_velocity must be provided for TARGET_VELOCITY strategy")
            return min(solutions, key=lambda s: abs(s.v0 - target_velocity))
        
        elif strategy == OptimizationStrategy.CUSTOM:
            if custom_scorer is None:
                raise ValueError("custom_scorer must be provided for CUSTOM strategy")
            return min(solutions, key=custom_scorer)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class PreciseBallisticCalculator(BallisticCalculator3D):
    """空気抵抗とマグヌス効果を考慮した精密計算"""
    
    def __init__(self, g=9.81, air_density=1.225, drag_coef=0.45, 
                 ball_radius=0.02, ball_mass=0.0027, magnus_coef=0.25):
        """
        Args:
            g: 重力加速度 (m/s^2)
            air_density: 空気密度 (kg/m^3)
            drag_coef: 抗力係数
            ball_radius: ボール半径 (m) - 卓球は40mm直径
            ball_mass: ボール質量 (kg) - 卓球は2.7g
            magnus_coef: マグヌス効果の係数
        """
        super().__init__(g)
        self.rho = air_density
        self.Cd = drag_coef
        self.radius = ball_radius
        self.mass = ball_mass
        self.A = np.pi * ball_radius**2  # 断面積
        self.Cm = magnus_coef
        
    def simulate_with_drag(self, v0: float, azimuth: float, elevation: float, 
                          spin_rate: float = 0, spin_axis: Tuple[float, float, float] = (0, 0, 1),
                          start_z: float = 0.0, dt: float = 0.001, max_time: float = 10.0) -> List[Point3D]:
        """
        空気抵抗とマグヌス効果を考慮した軌道計算（RK4法）
        
        Args:
            v0: 初速度 (m/s)
            azimuth: 方位角 (rad)
            elevation: 仰角 (rad)
            spin_rate: スピン速度 (rad/s)
            spin_axis: スピン軸の方向ベクトル (正規化不要)
            start_z: 初期高さ (m)
            dt: 時間刻み (s)
            max_time: 最大計算時間 (s)
        
        Returns:
            軌道点のリスト
        """
        # 初期速度ベクトル
        vx = v0 * np.cos(elevation) * np.cos(azimuth)
        vy = v0 * np.cos(elevation) * np.sin(azimuth)
        vz = v0 * np.sin(elevation)
        
        # 初期位置
        x, y, z = 0.0, 0.0, start_z
        
        # スピン軸の正規化
        spin_norm = np.linalg.norm(spin_axis)
        if spin_norm > 0:
            omega_vec = np.array(spin_axis) * spin_rate / spin_norm
        else:
            omega_vec = np.array([0.0, 0.0, 0.0])
        
        trajectory = []
        t = 0.0
        
        def acceleration(vel):
            """加速度を計算"""
            v_mag = np.linalg.norm(vel)
            
            if v_mag < 1e-6:
                return np.array([0.0, 0.0, -self.g])
            
            # 空気抵抗
            drag_force = -0.5 * self.rho * self.Cd * self.A * v_mag
            a_drag = drag_force * vel / self.mass
            
            # マグヌス力 F = Cm * rho * A * r * omega × v
            magnus_force = self.Cm * self.rho * self.A * self.radius * np.cross(omega_vec, vel)
            a_magnus = magnus_force / self.mass
            
            # 重力
            a_gravity = np.array([0.0, 0.0, -self.g])
            
            return a_drag + a_magnus + a_gravity
        
        while z >= 0 and t < max_time:
            trajectory.append(Point3D(x, y, z))
            
            # RK4法による数値積分
            pos = np.array([x, y, z])
            vel = np.array([vx, vy, vz])
            
            k1_v = acceleration(vel)
            k1_p = vel
            
            k2_v = acceleration(vel + 0.5 * dt * k1_v)
            k2_p = vel + 0.5 * dt * k1_v
            
            k3_v = acceleration(vel + 0.5 * dt * k2_v)
            k3_p = vel + 0.5 * dt * k2_v
            
            k4_v = acceleration(vel + dt * k3_v)
            k4_p = vel + dt * k3_v
            
            vel += dt * (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            pos += dt * (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
            
            vx, vy, vz = vel
            x, y, z = pos
            t += dt
        
        return trajectory
    
    def optimize_parameters(self, target: Point3D, start_z: float, 
                          initial_guess: Tuple[float, float, float],
                          spin_rate: float = 0,
                          spin_axis: Tuple[float, float, float] = (0, 0, 1),
                          method: str = 'Nelder-Mead',
                          tolerance: float = 1e-4) -> dict:
        """
        シンプルモデルの解を初期値として最適化
        
        Args:
            target: 目標位置
            start_z: 開始高さ
            initial_guess: 初期推定値 (v0, azimuth_rad, elevation_rad)
            spin_rate: スピン速度 (rad/s)
            spin_axis: スピン軸
            method: 最適化手法 ('Nelder-Mead', 'Powell', 'BFGS' など)
            tolerance: 収束判定の閾値
        
        Returns:
            最適化結果の辞書
        """
        def objective(params):
            v0, az, el = params
            
            # パラメータの妥当性チェック
            if v0 <= 0:
                return 1e10
            if el < -np.pi/2 or el > np.pi/2:
                return 1e10
            
            trajectory = self.simulate_with_drag(v0, az, el, spin_rate, spin_axis, start_z=start_z)
            
            if not trajectory or len(trajectory) < 2:
                return 1e10
            
            # 最終点と目標点の距離
            final = trajectory[-1]
            error = np.sqrt(
                (final.x - target.x)**2 + 
                (final.y - target.y)**2 + 
                (final.z - target.z)**2
            )
            return error
        
        result = minimize(
            objective,
            initial_guess,
            method=method,
            options={'maxiter': 1000, 'xatol': tolerance, 'fatol': tolerance}
        )
        
        if result.success:
            v0_opt, az_opt, el_opt = result.x
            final_trajectory = self.simulate_with_drag(v0_opt, az_opt, el_opt, 
                                                       spin_rate, spin_axis, start_z=start_z)
        else:
            final_trajectory = []
        
        return {
            'success': result.success,
            'v0': result.x[0],
            'azimuth_rad': result.x[1],
            'elevation_rad': result.x[2],
            'azimuth_deg': result.x[1] * 180 / np.pi,
            'elevation_deg': result.x[2] * 180 / np.pi,
            'error': result.fun,
            'trajectory': final_trajectory,
            'iterations': result.nit if hasattr(result, 'nit') else None,
            'message': result.message
        }


def find_precise_solution(target: Point3D, start_z: float, velocities: List[float],
                         strategy: OptimizationStrategy = OptimizationStrategy.MIN_ELEVATION,
                         target_elevation: Optional[float] = None,
                         target_velocity: Optional[float] = None,
                         custom_scorer: Optional[Callable[[Solution], float]] = None,
                         spin_rate: float = 0,
                         spin_axis: Tuple[float, float, float] = (0, 0, 1),
                         verbose: bool = True,
                         allow_neg_elev: bool = True) -> Optional[dict]:
    """
    シンプルモデルで初期解を求め、精密モデルで最適化
    
    Args:
        target: 目標位置
        start_z: 開始高さ
        velocities: テストする速度のリスト
        strategy: 初期解選択戦略
        target_elevation: 目標仰角（度）
        target_velocity: 目標速度（m/s）
        custom_scorer: カスタム評価関数
        spin_rate: スピン速度 (rad/s)
        spin_axis: スピン軸
        verbose: 詳細出力
    
    Returns:
        精密解の辞書、または None
    """
    # Step 1: シンプルモデルで初期解を取得
    simple_calc = BallisticCalculator3D()
    simple_solutions = simple_calc.find_solutions(velocities, target, start_z)
    
    if not simple_solutions:
        if verbose:
            print("❌ シンプルモデルでも解が見つかりません")
        return None
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 シンプルモデル: {len(simple_solutions)}個の解が見つかりました")
        print(f"{'='*70}")
    
    # Step 2: 戦略に基づいて最適な初期解を選択
    best_simple = simple_calc.select_best_solution(
        simple_solutions,
        strategy=strategy,
        target_elevation=target_elevation,
        target_velocity=target_velocity,
        custom_scorer=custom_scorer
    )
    
    if verbose:
        print(f"\n🎯 選択された初期解 (戦略: {strategy.value}):")
        print(f"  速度:   {best_simple.v0:.2f} m/s")
        print(f"  方位角: {best_simple.azimuth:.2f}°")
        print(f"  仰角:   {best_simple.elevation:.2f}°")
        print(f"  合計角度: {best_simple.total_angle:.2f}°")
    
    # Step 3: 精密モデルで最適化
    if verbose:
        print(f"\n{'='*70}")
        print("🔬 精密モデルで最適化中...")
        print(f"{'='*70}")
    
    precise_calc = PreciseBallisticCalculator()
    initial_guess = [
        best_simple.v0,
        best_simple.azimuth * np.pi / 180,
        best_simple.elevation * np.pi / 180
    ]
    
    result = precise_calc.optimize_parameters(
        target, start_z, initial_guess,
        spin_rate=spin_rate,
        spin_axis=spin_axis
    )
    
    if verbose:
        print(f"\n{'='*70}")
        if result['success']:
            print("✅ 最適化成功!")
            print(f"{'='*70}")
            print(f"\n📈 精密解:")
            print(f"  速度:   {result['v0']:.2f} m/s (初期値から {result['v0']-best_simple.v0:+.2f} m/s)")
            print(f"  方位角: {result['azimuth_deg']:.2f}° (初期値から {result['azimuth_deg']-best_simple.azimuth:+.2f}°)")
            print(f"  仰角:   {result['elevation_deg']:.2f}° (初期値から {result['elevation_deg']-best_simple.elevation:+.2f}°)")
            print(f"  誤差:   {result['error']*1000:.2f} mm")
            if result['iterations']:
                print(f"  反復回数: {result['iterations']}")
        else:
            print("❌ 最適化失敗")
            print(f"  メッセージ: {result['message']}")
        print(f"{'='*70}\n")
    
    # 結果に初期解も含める
    result['initial_solution'] = best_simple
    
    return result if result['success'] else None


def plot_comparison(simple_solution: Solution, precise_result: dict, 
                   target: Point3D, start_z: float):
    """シンプルモデルと精密モデルの比較プロット"""
    fig = plt.figure(figsize=(18, 6))
    
    # 3Dプロット
    ax1 = fig.add_subplot(131, projection='3d')
    
    # シンプルモデルの軌道
    x_simple = [p.x for p in simple_solution.trajectory]
    y_simple = [p.y for p in simple_solution.trajectory]
    z_simple = [p.z for p in simple_solution.trajectory]
    ax1.plot(x_simple, y_simple, z_simple, 'b-', linewidth=2, label='Simple Model')
    
    # 精密モデルの軌道
    if precise_result['trajectory']:
        x_precise = [p.x for p in precise_result['trajectory']]
        y_precise = [p.y for p in precise_result['trajectory']]
        z_precise = [p.z for p in precise_result['trajectory']]
        ax1.plot(x_precise, y_precise, z_precise, 'r-', linewidth=2, label='Precise Model')
    
    ax1.scatter([0], [0], [start_z], c='green', s=100, label='Start', depthshade=True)
    ax1.scatter([target.x], [target.y], [target.z], c='orange', s=100, label='Target', depthshade=True)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    
    # XY平面（上面図）
    ax2 = fig.add_subplot(132)
    ax2.plot(x_simple, y_simple, 'b-', linewidth=2, label='Simple Model')
    if precise_result['trajectory']:
        ax2.plot(x_precise, y_precise, 'r-', linewidth=2, label='Precise Model')
    ax2.plot(0, 0, 'go', markersize=10, label='Start')
    ax2.plot(target.x, target.y, 'o', color='orange', markersize=10, label='Target')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (XY plane)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    # 側面図
    ax3 = fig.add_subplot(133)
    h_simple = [np.sqrt(p.x**2 + p.y**2) for p in simple_solution.trajectory]
    ax3.plot(h_simple, z_simple, 'b-', linewidth=2, label='Simple Model')
    if precise_result['trajectory']:
        h_precise = [np.sqrt(p.x**2 + p.y**2) for p in precise_result['trajectory']]
        ax3.plot(h_precise, z_precise, 'r-', linewidth=2, label='Precise Model')
    
    target_dist = np.sqrt(target.x**2 + target.y**2)
    ax3.plot(0, start_z, 'go', markersize=10, label='Start')
    ax3.plot(target_dist, target.z, 'o', color='orange', markersize=10, label='Target')
    ax3.set_xlabel('Horizontal Distance (m)')
    ax3.set_ylabel('Height Z (m)')
    ax3.set_title('Side View')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    print("🏓 卓球ロボット - 弾道計算システム")
    print("="*70)
    
    # パラメータ設定
    start_z = 0.3
    target = Point3D(x=2.74/2, y=1.525, z=0)  # 卓球台の反対側中央くらい
    velocities = [5,10,15]
    
    print(f"\n📍 設定:")
    print(f"  ロボット位置: X=0m, Y=0m, Z={start_z}m")
    print(f"  目標位置: X={target.x}m, Y={target.y}m, Z={target.z}m")
    print(f"  水平距離: {np.sqrt(target.x**2 + target.y**2):.3f}m")
    print(f"  テスト速度: {velocities}")
    
    # 例1: 最小仰角を優先
    print("\n" + "="*70)
    print("例1: 最小仰角を優先")
    print("="*70)
    result1 = find_precise_solution(
        target, start_z, velocities,
        strategy=OptimizationStrategy.MIN_ELEVATION,
        spin_rate=1000,  # 100 rad/s のバックスピン
        spin_axis=(0, 0, 1)  # X軸周りの回転
    )
    
    if result1:
        plot_comparison(result1['initial_solution'], result1, target, start_z)
    
    # 例2: 目標仰角に近づける
    print("\n" + "="*70)
    print("例2: 目標仰角30度に近づける")
    print("="*70)
    result2 = find_precise_solution(
        target, start_z, velocities,
        strategy=OptimizationStrategy.TARGET_ANGLE,
        target_elevation=80.0,
        spin_rate=500,
        spin_axis=(0, 0, 1)
    )
    
    if result2:
        plot_comparison(result2['initial_solution'], result2, target, start_z)
    
    # 例3: カスタム評価関数（速度を重視しつつ角度も考慮）
    print("\n" + "="*70)
    print("例3: 速度が速く、かつ角度が小さい解を優先")
    print("="*70)
    
    def custom_scorer(sol: Solution) -> float:
        # 速度が速いほど良い（負の値）、角度が小さいほど良い
        # 重み付けで調整
        velocity_score = -sol.v0 * 0.1  # 速度10m/sで-1.0
        angle_score = sol.elevation * 1.0  # 角度そのまま
        return velocity_score + angle_score
    
    result3 = find_precise_solution(
        target, start_z, velocities,
        strategy=OptimizationStrategy.CUSTOM,
        custom_scorer=custom_scorer
    )
    
    if result3:
        plot_comparison(result3['initial_solution'], result3, target, start_z)


if __name__ == "__main__":
    main()