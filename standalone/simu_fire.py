import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

angle = -30.0
power = 0.7
spin = 50.0
spin_axis = (1.0, 0.0, 0.0)

launch_height = 10

g = 9.81
m = 0.145
rho = 1.225
C_d = 0.47
r = 0.0366
A = np.pi * r**2
dt = 0.001

e_normal = 0.6
e_tangential = 0.7
friction_coef = 0.3

v0 = power * 50

angle_rad = np.radians(angle)
vx = v0 * np.cos(angle_rad)
vy = 0.0
vz = v0 * np.sin(angle_rad)

spin_axis_norm = np.array(spin_axis) / np.linalg.norm(spin_axis)
omega = spin * np.array(spin_axis_norm)

pos = np.array([0.0, 0.0, launch_height])
vel = np.array([vx, vy, vz])

trajectory = [pos.copy()]
bounced = False
bounce_positions = []

max_iterations = 100000
iteration = 0

while iteration < max_iterations:
    iteration += 1
    v_mag = np.linalg.norm(vel)
    
    if v_mag > 0:
        drag = -0.5 * rho * C_d * A * v_mag * vel
        
        S = 4.1e-4
        v_cross_omega = np.cross(vel, omega)
        magnus = rho * S * v_cross_omega
        
        acc = (drag + magnus) / m + np.array([0, 0, -g])
    else:
        acc = np.array([0, 0, -g])
    
    vel += acc * dt
    pos += vel * dt
    
    if pos[2] < 0:
        if not bounced:
            pos[2] = 0
            
            v_normal = vel[2]
            v_tangential = np.array([vel[0], vel[1], 0])
            v_tangential_mag = np.linalg.norm(v_tangential)
            
            vel[2] = -v_normal * e_normal
            
            if v_tangential_mag > 0:
                friction_impulse = min(friction_coef * abs(v_normal), v_tangential_mag * e_tangential)
                vel[0] -= (vel[0] / v_tangential_mag) * friction_impulse
                vel[1] -= (vel[1] / v_tangential_mag) * friction_impulse
            
            omega *= 0.8
            
            bounce_positions.append(pos.copy())
            bounced = True
        else:
            break
    
    trajectory.append(pos.copy())
    
    if bounced and pos[2] < 0:
        break
    
    if bounced and v_mag < 0.1:
        break

trajectory = np.array(trajectory)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

if len(trajectory) > 0:
    if len(bounce_positions) > 0:
        bounce_idx = 0
        for i in range(len(trajectory) - 1):
            if np.allclose(trajectory[i], bounce_positions[0], atol=0.01):
                bounce_idx = i
                break
        
        ax.plot(trajectory[:bounce_idx+1, 0], trajectory[:bounce_idx+1, 1], 
                trajectory[:bounce_idx+1, 2], 'b-', linewidth=2, label='Before Bounce')
        ax.plot(trajectory[bounce_idx:, 0], trajectory[bounce_idx:, 1], 
                trajectory[bounce_idx:, 2], 'r-', linewidth=2, label='After Bounce')
    else:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, label='Trajectory')

ax.scatter([0], [0], [launch_height], color='red', s=100, label='Start', marker='o')

if len(bounce_positions) > 0:
    for bp in bounce_positions:
        ax.scatter([bp[0]], [bp[1]], [bp[2]], color='orange', s=150, 
                   label='Bounce' if bp is bounce_positions[0] else '', marker='*')

ax.scatter([trajectory[-1, 0]], [trajectory[-1, 1]], [trajectory[-1, 2]], 
           color='green', s=100, label='End', marker='s')

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.set_title(f'Ball Trajectory with Bounce\nAngle: {angle}°, Power: {power}, Spin: {spin} rad/s', 
             fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

max_range = max(np.max(np.abs(trajectory[:, 0])), 
                np.max(np.abs(trajectory[:, 1])),
                np.max(trajectory[:, 2])) if len(trajectory) > 0 else 10
ax.set_xlim([0, max_range * 1.1])
ax.set_ylim([-max_range * 0.5, max_range * 0.5])
ax.set_zlim([0, max_range * 0.6])

distance = np.sqrt(trajectory[-1, 0]**2 + trajectory[-1, 1]**2)
max_height = np.max(trajectory[:, 2])
flight_time = len(trajectory) * dt

info_text = f'Distance: {distance:.2f} m\nMax Height: {max_height:.2f} m\nFlight Time: {flight_time:.2f} s'
if len(bounce_positions) > 0:
    info_text += f'\nBounces: {len(bounce_positions)}'
ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
          fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()