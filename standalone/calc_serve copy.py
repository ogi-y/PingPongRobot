import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

def calc_impact_point(level, oppo_pos, court_size=(2.74, 1.525)):
    table_length, table_width = court_size
    params = {
        0: {"avoid": 0.0, "random": 0.1, "depth_range": [1]},
        1: {"avoid": 0.3, "random": 0.15, "depth_range": [0, 1, 2]},
        2: {"avoid": 0.7, "random": 0.2, "depth_range": [0, 1, 2]},
        3: {"avoid": 1.0, "random": 0.1, "depth_range": [0, 1, 2]},
    }
    p = params.get(level, params[1])

    grid_y_start = table_length / 2
    grid_y_size = table_length / 2 / 3
    grid_x_size = table_width / 3

    oppo_grid_y = int((oppo_pos[1] - grid_y_start) / grid_y_size)
    oppo_grid_x = int((oppo_pos[0] + table_width / 2) / grid_x_size)
    oppo_grid_y = np.clip(oppo_grid_y, 0, 2)
    oppo_grid_x = np.clip(oppo_grid_x, 0, 2)

    if np.random.rand() < p["avoid"]:
        candidates_x = [j for j in range(3) if j != oppo_grid_x]
        chosen_grid_x = np.random.choice(candidates_x)
    else:
        chosen_grid_x = oppo_grid_x

    if np.random.rand() < p["avoid"]:
        candidates_y = [i for i in p["depth_range"] if i != oppo_grid_y]
        if len(candidates_y) > 0:
            chosen_grid_y = np.random.choice(candidates_y)
        else:
            chosen_grid_y = oppo_grid_y
    else:
        if oppo_grid_y in p["depth_range"]:
            chosen_grid_y = oppo_grid_y
        else:
            chosen_grid_y = np.random.choice(p["depth_range"])

    center_x = -table_width / 2 + (chosen_grid_x + 0.5) * grid_x_size
    center_y = grid_y_start + (chosen_grid_y + 0.5) * grid_y_size

    offset_x = np.random.uniform(-p["random"] * grid_x_size, p["random"] * grid_x_size)
    offset_y = np.random.uniform(-p["random"] * grid_y_size, p["random"] * grid_y_size)

    target_x = np.clip(center_x + offset_x, -table_width / 2 + 0.05, table_width / 2 - 0.05)
    target_y = np.clip(center_y + offset_y, grid_y_start + 0.05, table_length - 0.05)

    return target_x, target_y

def calc_launch_params(robo_pos, target_pos, level, court_size=(2.74, 1.525), net_height=0.1525):
    table_length, table_width = court_size
    
    level_params = {
        0: {"power_scale": 0.5, "max_spin_rpm": 50},
        1: {"power_scale": 0.7, "max_spin_rpm": 150},
        2: {"power_scale": 0.9, "max_spin_rpm": 250},
        3: {"power_scale": 1.0, "max_spin_rpm": 350},
    }
    p = level_params.get(level, level_params[1])
    
    dx = target_pos[0] - robo_pos[0]
    dy = target_pos[1] - robo_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    net_y = table_length / 2
    net_clearance = 0.05
    
    if robo_pos[1] < net_y < target_pos[1]:
        dist_to_net = net_y - robo_pos[1]
        min_angle = np.arctan((net_height + net_clearance) / dist_to_net)
    else:
        min_angle = np.radians(5)
    
    launch_angle = min_angle + np.radians(3 + level * 2)
    launch_angle = np.clip(launch_angle, np.radians(5), np.radians(30))
    
    g = 9.81
    v0_base = np.sqrt(g * distance / np.sin(2 * launch_angle))
    v0 = v0_base * p["power_scale"]
    
    azimuth = np.arctan2(dy, dx)
    
    spin_rpm = p["max_spin_rpm"]
    
    return v0, launch_angle, azimuth, spin_rpm

def calc_motor_params(launch_params, wheel_radius=0.02, wheel_separation=0.05):
    v0, launch_angle, azimuth, spin_rpm = launch_params
    
    pitch = launch_angle
    yaw = azimuth
    roll = 0.0
    
    base_rpm = (v0 / (2 * np.pi * wheel_radius)) * 60
    
    spin_diff = spin_rpm * 0.3
    
    shoot_left = base_rpm + spin_diff
    shoot_right = base_rpm - spin_diff
    
    return roll, pitch, yaw, shoot_left, shoot_right

table_length = 2.74
table_width = 1.525
fig = plt.figure(figsize=(14, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
plt.subplots_adjust(bottom=0.35, wspace=0.3)

def setup_court(ax):
    rect = plt.Rectangle((-table_width / 2, 0), table_width, table_length, linewidth=2, edgecolor='blue', facecolor='lightblue')
    ax.add_patch(rect)
    ax.plot([-table_width / 2, table_width / 2], [table_length / 2, table_length / 2], color='white', linewidth=2)
    ax.plot([-table_width / 2, table_width / 2], [table_length / 2, table_length / 2], color='black', linestyle='--', linewidth=1)
    
    for side in [0, table_length / 2]:
        for i in range(1, 3):
            y = side + (table_length / 2) * i / 3
            ax.plot([-table_width / 2, table_width / 2], [y, y], color='gray', linestyle='--', linewidth=1)
        for i in range(1, 3):
            x = -table_width / 2 + table_width * i / 3
            ax.plot([x, x], [side, side + table_length / 2], color='gray', linestyle='--', linewidth=1)
    
    ax.set_xlim(-table_width / 2 - 0.2, table_width / 2 + 0.2)
    ax.set_ylim(-0.2, table_length + 1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

setup_court(ax1)
ax1.set_title('Single Shot')
setup_court(ax2)
ax2.set_title('Heatmap (1000 shots)')

robo_pos = (0, 0)
oppo_pos = (0, table_length + 0.5)

robo_plot1, = ax1.plot(robo_pos[0], robo_pos[1], 'ro', markersize=12, label='Robot')
oppo_plot1, = ax1.plot(oppo_pos[0], oppo_pos[1], 'bo', markersize=12, label='Opponent')
serve_line, = ax1.plot([], [], 'g--', linewidth=2, label='Serve Trajectory')
target_plot, = ax1.plot([], [], 'gx', markersize=10, label='Target Point')

robo_plot2, = ax2.plot(robo_pos[0], robo_pos[1], 'ro', markersize=12, label='Robot')
oppo_plot2, = ax2.plot(oppo_pos[0], oppo_pos[1], 'bo', markersize=12, label='Opponent')
heatmap_data = None

axcolor = 'lightgoldenrodyellow'
ax_robo_x = plt.axes([0.15, 0.13, 0.3, 0.03], facecolor=axcolor)
ax_robo_y = plt.axes([0.15, 0.18, 0.3, 0.03], facecolor=axcolor)
slider_robo_x = Slider(ax_robo_x, 'Robot X', -table_width / 2, table_width / 2, valinit=robo_pos[0])
slider_robo_y = Slider(ax_robo_y, 'Robot Y', 0, table_length, valinit=robo_pos[1])
ax_oppo_x = plt.axes([0.15, 0.08, 0.3, 0.03], facecolor=axcolor)
ax_oppo_y = plt.axes([0.15, 0.03, 0.3, 0.03], facecolor=axcolor)
slider_oppo_x = Slider(ax_oppo_x, 'Opponent X', -table_width / 2, table_width / 2, valinit=oppo_pos[0])
slider_oppo_y = Slider(ax_oppo_y, 'Opponent Y', table_length, table_length + 1.0, valinit=oppo_pos[1])
slider_level = Slider(plt.axes([0.15, 0.23, 0.3, 0.03], facecolor=axcolor), 'Level', 0, 3, valinit=1, valstep=1)

ax_fire = plt.axes([0.55, 0.2, 0.1, 0.04])
btn_fire = Button(ax_fire, 'Fire!', color=axcolor, hovercolor='0.975')
ax_heatmap = plt.axes([0.70, 0.2, 0.15, 0.04])
btn_heatmap = Button(ax_heatmap, 'Generate Heatmap', color=axcolor, hovercolor='0.975')

param_text = ax1.text(2, 0.9, '', transform=ax1.transAxes, 
                      verticalalignment='top', fontfamily='monospace', fontsize=8)

def update(val):
    robo_plot1.set_data([slider_robo_x.val], [slider_robo_y.val])
    oppo_plot1.set_data([slider_oppo_x.val], [slider_oppo_y.val])
    robo_plot2.set_data([slider_robo_x.val], [slider_robo_y.val])
    oppo_plot2.set_data([slider_oppo_x.val], [slider_oppo_y.val])
    fig.canvas.draw_idle()

def fire(event):
    robo_x = slider_robo_x.val
    robo_y = slider_robo_y.val
    oppo_x = slider_oppo_x.val
    oppo_y = slider_oppo_y.val
    level = int(slider_level.val)
    
    target_x, target_y = calc_impact_point(level=level, oppo_pos=(oppo_x, oppo_y), 
                                           court_size=(table_length, table_width))
    
    launch_params = calc_launch_params((robo_x, robo_y), (target_x, target_y), level,
                                       court_size=(table_length, table_width))
    
    motor_params = calc_motor_params(launch_params)
    
    ext_x = target_x + (target_x - robo_x) * 0.5
    ext_y = target_y + (target_y - robo_y) * 0.5
    serve_line.set_data([robo_x, ext_x], [robo_y, ext_y])
    target_plot.set_data([target_x], [target_y])
    
    v0, angle, azimuth, spin = launch_params
    roll, pitch, yaw, shoot_l, shoot_r = motor_params
    
    info_text = f'Target: ({target_x:.3f}, {target_y:.3f})\n'
    info_text += f'V0: {v0:.2f} m/s\n'
    info_text += f'Angle: {np.degrees(angle):.1f}°\n'
    info_text += f'Azimuth: {np.degrees(azimuth):.1f}°\n'
    info_text += f'Spin: {spin:.0f} rpm\n'
    info_text += f'Motor L: {shoot_l:.0f} rpm\n'
    info_text += f'Motor R: {shoot_r:.0f} rpm'
    param_text.set_text(info_text)
    
    fig.canvas.draw_idle()

def generate_heatmap(event):
    global heatmap_data
    
    oppo_x = slider_oppo_x.val
    oppo_y = slider_oppo_y.val
    level = int(slider_level.val)
    
    targets_x = []
    targets_y = []
    for _ in range(1000):
        tx, ty = calc_impact_point(level=level, oppo_pos=(oppo_x, oppo_y), 
                                   court_size=(table_length, table_width))
        targets_x.append(tx)
        targets_y.append(ty)
    
    if heatmap_data is not None:
        heatmap_data.remove()
    
    heatmap_data = ax2.hexbin(targets_x, targets_y, gridsize=15, cmap='YlOrRd', 
                               extent=(-table_width/2, table_width/2, table_length/2, table_length),
                               alpha=0.7, mincnt=1)
    
    if not hasattr(generate_heatmap, 'colorbar_added'):
        plt.colorbar(heatmap_data, ax=ax2, label='Shot Count')
        generate_heatmap.colorbar_added = True
    
    fig.canvas.draw_idle()

btn_fire.on_clicked(fire)
btn_heatmap.on_clicked(generate_heatmap)
slider_robo_x.on_changed(update)
slider_robo_y.on_changed(update)
slider_oppo_x.on_changed(update)
slider_oppo_y.on_changed(update)

plt.show()