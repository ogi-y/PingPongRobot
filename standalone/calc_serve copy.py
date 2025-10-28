import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 卓球台（XZ平面）
table_length = 2.74
table_width = 1.525
table_height = 0  # 卓球台の高さ（m）
# robo
robo_pos = np.array([-0.3, table_width/2, 0.2])  # ロボットの位置
# opponent
oppo_pos = np.array([3.0, 1.0, 0.2])  # 相手の位置
# start and target
start_pos = robo_pos
target_pos = np.array([2.4, 1.2, 0.76])  # サーブの目標位置

# 台の四隅
x = [0, table_length, table_length, 0, 0]
y = [0, 0, table_width, table_width, 0]
z = [0, 0, 0, 0, 0]
ax.plot(x, y, z, color='green', linewidth=2)

# ネット
net_x = [table_length/2, table_length/2]
net_y = [0, table_width]
net_z = [0, 0.1525]  # ネットの高さ
ax.plot(net_x, net_y, [0, 0], color='black', linewidth=3)
ax.plot(net_x, net_y, [0.1525, 0.1525], color='black', linewidth=3)

# サーブ軌道（例：放物線）
start = start_pos
target = target_pos
t = np.linspace(0, 1, 30)
traj = start[None, :] + (target - start)[None, :] * t[:, None]
# 軌道に高さ変化を追加（簡易放物線）
traj[:,2] += 0.2 * np.sin(np.pi * t)
ax.plot(traj[:,0], traj[:,1], traj[:,2], '--', color='orange', linewidth=2)

# 相手
ax.scatter(oppo_pos[0], oppo_pos[1], oppo_pos[2], color='blue', s=100)
# robo
ax.scatter(robo_pos[0], robo_pos[1], robo_pos[2], color='red', s=100)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Serve Trajectory')
ax.set_xlim(-0.5, 3.2)
ax.set_ylim(-0.5, table_width + 0.5)
ax.set_zlim(0, 1.2)
plt.tight_layout()
plt.show()