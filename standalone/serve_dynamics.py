import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
                      start_z: float = 0.0) -> List[Solution]:
        solutions = []
        max_dist = np.sqrt(target.x**2 + target.y**2)
        
        for v0 in velocities:
            angles = self.solve_angles(v0, target, start_z)
            
            if angles is None:
                continue
            
            azimuth_high, elev_high, azimuth_low, elev_low = angles
            
            elev_high_deg = elev_high * 180 / np.pi
            elev_low_deg = elev_low * 180 / np.pi
            
            if 0 <= elev_high_deg <= 90:
                trajectory = self.calculate_trajectory(v0, azimuth_high, elev_high, start_z, max_dist)
                solutions.append(Solution(
                    v0=v0,
                    azimuth=azimuth_high * 180 / np.pi,
                    elevation=elev_high_deg,
                    trajectory=trajectory
                ))
            
            if 0 <= elev_low_deg <= 90 and abs(elev_high_deg - elev_low_deg) > 0.1:
                trajectory = self.calculate_trajectory(v0, azimuth_low, elev_low, start_z, max_dist)
                solutions.append(Solution(
                    v0=v0,
                    azimuth=azimuth_low * 180 / np.pi,
                    elevation=elev_low_deg,
                    trajectory=trajectory
                ))
        
        return solutions

def plot_trajectories_3d(solutions: List[Solution], target: Point3D, start_z: float = 0.0):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(solutions)))
    
    for idx, sol in enumerate(solutions):
        x_coords = [p.x for p in sol.trajectory]
        y_coords = [p.y for p in sol.trajectory]
        z_coords = [p.z for p in sol.trajectory]
        
        traj_type = 'high' if sol.elevation > 45 else 'low'
        label = f'v={sol.v0}m/s ({traj_type} el={sol.elevation:.1f}° az={sol.azimuth:.1f}°)'
        ax.plot(x_coords, y_coords, z_coords, '-', color=colors[idx], label=label, linewidth=2)
    
    ax.scatter([0], [0], [start_z], c='green', s=100, label='Start', depthshade=True)
    ax.scatter([target.x], [target.y], [target.z], c='red', s=100, label='Target', depthshade=True)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('3D Ballistic Trajectories', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    max_range = max(abs(target.x), abs(target.y), np.sqrt(target.x**2 + target.y**2))
    ax.set_xlim([-max_range*0.1, max_range*1.1])
    ax.set_ylim([-max_range*0.1, max_range*1.1])
    ax.set_zlim([0, max(start_z, target.z) * 1.5])
    
    plt.tight_layout()
    plt.show()

def plot_trajectories_2d(solutions: List[Solution], target: Point3D, start_z: float = 0.0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(solutions)))
    
    for idx, sol in enumerate(solutions):
        x_coords = [p.x for p in sol.trajectory]
        y_coords = [p.y for p in sol.trajectory]
        z_coords = [p.z for p in sol.trajectory]
        horizontal = [np.sqrt(p.x**2 + p.y**2) for p in sol.trajectory]
        
        traj_type = 'high' if sol.elevation > 45 else 'low'
        label = f'v={sol.v0}m/s ({traj_type} el={sol.elevation:.1f}°)'
        
        ax1.plot(x_coords, y_coords, '-', color=colors[idx], label=label, linewidth=2)
        ax2.plot(horizontal, z_coords, '-', color=colors[idx], label=label, linewidth=2)
    
    ax1.plot(0, 0, 'go', markersize=10, label='Start')
    ax1.plot(target.x, target.y, 'ro', markersize=10, label='Target')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_title('Top View (XY plane)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')
    
    target_dist = np.sqrt(target.x**2 + target.y**2)
    ax2.plot(0, start_z, 'go', markersize=10, label='Start')
    ax2.plot(target_dist, target.z, 'ro', markersize=10, label='Target')
    ax2.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax2.set_ylabel('Height Z (m)', fontsize=12)
    ax2.set_title('Side View (Distance-Z plane)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def print_solutions(solutions: List[Solution]):
    if not solutions:
        print("No solutions found (velocity may be insufficient)")
        return
    
    print("\n=== Solutions ===")
    print(f"{'Velocity (m/s)':<15} {'Azimuth (deg)':<15} {'Elevation (deg)':<18} {'Type':<10}")
    print("-" * 70)
    
    for sol in solutions:
        traj_type = 'high' if sol.elevation > 45 else 'low'
        print(f"{sol.v0:<15.2f} {sol.azimuth:<15.2f} {sol.elevation:<18.2f} {traj_type:<10}")

def main():
    calculator = BallisticCalculator3D(g=9.81)
    
    start_z = 0.2
    target = Point3D(x=5.0, y=8.0, z=0.0)
    velocities = [8, 10, 12, 15, 18]
    
    print(f"Robot position: X=0m, Y=0m, Z={start_z}m (origin, adjustable height)")
    print(f"Target: X={target.x}m, Y={target.y}m, Z={target.z}m")
    print(f"Horizontal distance: {np.sqrt(target.x**2 + target.y**2):.2f}m")
    print(f"Testing velocities: {velocities}")
    
    solutions = calculator.find_solutions(velocities, target, start_z)
    
    print_solutions(solutions)
    
    if solutions:
        #plot_trajectories_3d(solutions, target, start_z)
        plot_trajectories_2d(solutions, target, start_z)

if __name__ == "__main__":
    main()