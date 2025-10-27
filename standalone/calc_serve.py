import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class PingPongVisualization:
    def __init__(self):
        # 卓球台のサイズ（メートル）
        self.table_length = 2.74
        self.table_width = 1.525
        self.net_position = self.table_length / 2
        
    def setup_court(self, ax):
        """卓球台を描画"""
        # 台全体
        table = patches.Rectangle((0, 0), self.table_length, self.table_width,
                                   linewidth=2, edgecolor='black', facecolor='darkgreen', alpha=0.3)
        ax.add_patch(table)
        
        # センターライン
        ax.plot([self.net_position, self.net_position], [0, self.table_width], 
                'k-', linewidth=3, label='net')
        
        # 自陣と相手陣の分割線（視覚的に）
        ax.plot([0, self.table_length], [self.table_width/2, self.table_width/2], 
                'k--', linewidth=0.5, alpha=0.5)
        
        # 台の外側も表示するために範囲を広げる
        ax.set_xlim(-0.5, self.table_length + 0.5)
        ax.set_ylim(-0.5, self.table_width + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('length (m)')
        ax.set_ylabel('width (m)')
        ax.grid(True, alpha=0.3)
        
    def draw_target_zone(self, ax, target_x, target_y, level, color='red'):
        """ターゲットゾーンを描画"""
        # レベルに応じた範囲
        radius_map = {1: 0.3, 2: 0.2, 3: 0.15, 4: 0.1, 5: 0.08}
        radius = radius_map.get(level, 0.15)
        
        circle = patches.Circle((target_x, target_y), radius, 
                                linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(circle)
        ax.plot(target_x, target_y, 'x', color=color, markersize=10, markeredgewidth=2)
        
    def draw_opponent(self, ax, opponent_x, opponent_y, color='blue'):
        """相手の位置を描画（台の外側）"""
        # 相手を円で表現
        opponent = patches.Circle((opponent_x, opponent_y), 0.15, 
                                  linewidth=2, edgecolor=color, facecolor=color, alpha=0.6)
        ax.add_patch(opponent)
        ax.text(opponent_x, opponent_y - 0.3, 'opponent', 
                ha='center', fontsize=10, weight='bold')
        
    def draw_serve_trajectory(self, ax, start_x, start_y, target_x, target_y, 
                              num_points=20, color='orange'):
        """サーブの軌道を描画（簡易版）"""
        x = np.linspace(start_x, target_x, num_points)
        y = np.linspace(start_y, target_y, num_points)

        ax.plot(x, y, '--', color=color, linewidth=2, alpha=0.7, label='predicted trajectory')
        ax.plot(target_x, target_y, 'o', color=color, markersize=8)
        
    def visualize_level(self, level, opponent_pos=None):
        """レベルに応じたサーブを可視化"""
        fig, ax = plt.subplots(figsize=(12, 8))
        self.setup_court(ax)
        
        # サーブ開始位置（自陣の中央後方、台の外）
        start_x = -0.3
        start_y = self.table_width / 2
        
        # レベルに応じたターゲット設定（相手陣の台上）
        level_configs = {
            1: {'target': (2.0, 0.76), 'color': 'green'},
            2: {'target': (2.2, 0.4), 'color': 'yellow'},
            3: {'target': (2.4, 1.2), 'color': 'orange'},
            4: {'target': (2.5, 0.3), 'color': 'red'},
            5: {'target': (2.6, 1.3), 'color': 'darkred'}
        }
        
        config = level_configs.get(level, level_configs[3])
        target_x, target_y = config['target']
        
        # 描画
        self.draw_target_zone(ax, target_x, target_y, level, config['color'])
        self.draw_serve_trajectory(ax, start_x, start_y, target_x, target_y, color=config['color'])
        
        # 相手の位置（台の外側、相手陣側）
        if opponent_pos is None:
            opponent_pos = (self.table_length + 0.3, self.table_width / 2)  # 台の外、相手側
        self.draw_opponent(ax, opponent_pos[0], opponent_pos[1])
        
        # ロボットの位置も表示
        robot = patches.Circle((start_x, start_y), 0.1, 
                              linewidth=2, edgecolor='red', facecolor='red', alpha=0.6)
        ax.add_patch(robot)
        ax.text(start_x, start_y - 0.3, 'robot', 
                ha='center', fontsize=9, weight='bold')
        
        ax.set_title(f'Level {level} - Serve Course', fontsize=14, weight='bold')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        return fig, ax

# 使用例
viz = PingPongVisualization()

# レベル3のサーブを可視化、相手は右寄りに配置
fig, ax = viz.visualize_level(level=3, opponent_pos=(3.0, 1.0))
plt.show()

# 複数レベルを比較
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, level in enumerate([1, 3, 5]):
    viz.setup_court(axes[i])
    
    level_configs = {
        1: {'target': (2.0, 0.76), 'color': 'green'},
        3: {'target': (2.4, 1.2), 'color': 'orange'},
        5: {'target': (2.6, 1.3), 'color': 'darkred'}
    }
    
    config = level_configs[level]
    target_x, target_y = config['target']
    
    viz.draw_target_zone(axes[i], target_x, target_y, level, config['color'])
    viz.draw_serve_trajectory(axes[i], -0.3, 0.76, target_x, target_y, color=config['color'])
    
    # 相手は台の外側
    viz.draw_opponent(axes[i], 3.0, 0.76)
    
    # ロボットも表示
    robot = patches.Circle((-0.3, 0.76), 0.1, 
                          linewidth=2, edgecolor='red', facecolor='red', alpha=0.6)
    axes[i].add_patch(robot)

    axes[i].set_title(f'Level {level}', fontsize=12, weight='bold')

plt.tight_layout()
plt.show()