import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

def cal_serve_cource(level, oppo_pos, court_size=(2.74, 1.525), net_height=0.15):
    """サーブの目標位置を計算（簡易版）"""
    table_length, table_width = court_size
    # レベルに応じたターゲットゾーンの範囲
    params = {
        0: {"avoid": 0.0, "spin": 0.1, "speed": 0.2, "random": 0.1},  # easy
        1: {"avoid": 0.2, "spin": 0.2, "speed": 0.3, "random": 0.3},  # medium
        2: {"avoid": 0.5, "spin": 0.3, "speed": 0.4, "random": 0.5}, # hard
        3: {"avoid": 0.8, "spin": 0.4, "speed": 0.5, "random": 0.7},  # impossible
        "Rebellion": {"avoid": 0.0, "spin": "top_only", "speed": "max", "random": 0.0}  # Rebellion
    }
    
    p = params.get(level, params[1])
    # 相手の位置に基づいてターゲット位置を決定
    target_x = np.clip(oppo_pos[0] - np.random.uniform(p["avoid"], p["avoid"] + 0.5), table_length/2, table_length - 0.1)
    target_y = np.clip(oppo_pos[1] + np.random.uniform(-p["random"], p["random"]), 0, table_width)
    return target_x, target_y

# 卓球台のサイズ（単位: m）
table_length = 2.74
table_width = 1.525
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.5)

# ロボ
robo_pos = (0, table_width / 2)
# 相手
oppo_pos = (table_length + 0.5, table_width / 2)

# 卓球台の枠
rect = plt.Rectangle((0, 0), table_length, table_width, linewidth=2, edgecolor='blue', facecolor='lightblue')
ax.add_patch(rect)

# センターライン
ax.plot([table_length/2, table_length/2], [0, table_width], color='white', linewidth=2)

# ネット
ax.plot([table_length/2, table_length/2], [0, table_width], color='black', linestyle='--', linewidth=2)

# 9分割線（左右コートそれぞれ）
for side in [0, table_length/2]:
    # 縦方向（幅方向）2本
    for i in range(1, 3):
        x = side + (table_length/2) * i / 3
        ax.plot([x, x], [0, table_width], color='gray', linestyle='--', linewidth=1)
    # 横方向（長さ方向）2本
    for i in range(1, 3):
        y = table_width * i / 3
        ax.plot([side, side + table_length/2], [y, y], color='gray', linestyle='--', linewidth=1)

# 軸の設定
ax.set_xlim(-table_length, table_length*2)
ax.set_ylim(-table_width, table_width*2)
ax.set_aspect('equal')
ax.set_title('Top View of Ping Pong Table')
ax.set_xlabel('Length (m)')
ax.set_ylabel('Width (m)')

robo_plot, = ax.plot(robo_pos[0], robo_pos[1], 'ro', markersize=12, label='Robot')
oppo_plot, = ax.plot(oppo_pos[0], oppo_pos[1], 'bo', markersize=12, label='Opponent')
serve_line, = ax.plot([], [], 'g--', linewidth=2, label='Serve Trajectory')
# ウィジェット
axcolor = 'lightgoldenrodyellow'
ax_robo_x = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_robo_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
slider_robo_x = Slider(ax_robo_x, 'Robot X', 0, table_length, valinit=robo_pos[0])
slider_robo_y = Slider(ax_robo_y, 'Robot Y', 0, table_width, valinit=robo_pos[1])
ax_oppo_x = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_oppo_y = plt.axes([0.25, 0.0, 0.65, 0.03], facecolor=axcolor)
slider_oppo_x = Slider(ax_oppo_x, 'Opponent X', table_length, table_length + 1.0, valinit=oppo_pos[0])
slider_oppo_y = Slider(ax_oppo_y, 'Opponent Y', 0, table_width, valinit=oppo_pos[1])

ax_fire = plt.axes([0.8, 0.35, 0.1, 0.04])
btn_fire = Button(ax_fire, 'Fire!', color=axcolor, hovercolor='0.975')

def update(val):
    robo_plot.set_data([slider_robo_x.val], [slider_robo_y.val])
    oppo_plot.set_data([slider_oppo_x.val], [slider_oppo_y.val])
    fig.canvas.draw_idle()

def fire(event):
    robo_x = slider_robo_x.val
    robo_y = slider_robo_y.val
    oppo_pos = (slider_oppo_x.val, slider_oppo_y.val)
    target_x, target_y = cal_serve_cource(level=1, oppo_pos=oppo_pos)
    ext_x = target_x + (target_x - robo_x) * 0.5
    ext_y = target_y + (target_y - robo_y) * 0.5
    serve_line.set_data([robo_x, ext_x], [robo_y, ext_y])
    fig.canvas.draw_idle()

btn_fire.on_clicked(fire)
slider_robo_x.on_changed(update)
slider_robo_y.on_changed(update)
slider_oppo_x.on_changed(update)
slider_oppo_y.on_changed(update)

plt.show()