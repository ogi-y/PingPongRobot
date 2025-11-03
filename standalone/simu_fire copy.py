import numpy as np 

g = 9.81

launch_positions = {
    0: {'x': 0, 'y': 0, 'z': 10.0, 'label': '左'},
    1: {'x': 20, 'y': 0, 'z': 10.0, 'label': '中央'},
    2: {'x': 40, 'y': 0, 'z': 10.0, 'label': '右'}
}

target_positions = {
    0: {'x': 10, 'y': 50, 'z': 0, 'label': '左上'},
    1: {'x': 20, 'y': 50, 'z': 0, 'label': '中央上'},
    2: {'x': 30, 'y': 50, 'z': 0, 'label': '右上'},
    3: {'x': 10, 'y': 30, 'z': 0, 'label': '左中'},
    4: {'x': 20, 'y': 30, 'z': 0, 'label': '中央'},
    5: {'x': 30, 'y': 30, 'z': 0, 'label': '右中'},
    6: {'x': 10, 'y': 10, 'z': 0, 'label': '左下'},
    7: {'x': 20, 'y': 10, 'z': 0, 'label': '中央下'},
    8: {'x': 30, 'y': 10, 'z': 0, 'label': '右下'}
}

def calc_azimuth_angle_3d(x_start, y_start, x_target, y_target):
    dx = x_target - x_start
    dy = y_target - y_start
    
    azimuth_rad = np.arctan2(dx, dy)
    azimuth_deg = np.rad2deg(azimuth_rad)
    
    return azimuth_deg

def calc_horizontal_distance(x_start, y_start, x_target, y_target):
    dx = x_target - x_start
    dy = y_target - y_start
    return np.sqrt(dx**2 + dy**2)

def calc_angle_for_target_2d(v0, z0, horizontal_dist, z_target):
    dx = horizontal_dist
    dy = z_target - z0
    
    v0_sq = v0 ** 2
    g_dx_sq = g * dx ** 2
    
    a = g_dx_sq
    b = -2 * v0_sq * dx
    c = g_dx_sq + 2 * v0_sq * dy
    
    disc = b**2 - 4*a*c
    
    if disc < 0:
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

def calc_min_velocity_3d(z0, horizontal_dist, z_target):
    dx = horizontal_dist
    dy = z_target - z0
    
    if dx == 0:
        return abs(dy) * np.sqrt(g / 2)
    
    discriminant = g * dx * (g * dx + 2 * dy)
    if discriminant < 0:
        v_min = np.sqrt(g * np.sqrt(dx**2 + dy**2))
    else:
        v_min = np.sqrt(0.5 * (g * dx + np.sqrt(discriminant)))
    
    return v_min

def fire(launch_pos, target_pos, v0=50.0):
    if launch_pos not in launch_positions:
        print(f"エラー: 発射位置 {launch_pos} は無効です（0-2を指定）")
        return None
    
    if target_pos not in target_positions:
        print(f"エラー: 着弾目標 {target_pos} は無効です（0-8を指定）")
        return None
    
    launch = launch_positions[launch_pos]
    target = target_positions[target_pos]
    
    print(f"\n=== fire({launch_pos}, {target_pos}) ===")
    print(f"発射位置: {launch['label']} ({launch['x']}, {launch['y']}, {launch['z']})")
    print(f"着弾目標: {target['label']} ({target['x']}, {target['y']}, {target['z']})")
    
    azimuth = calc_azimuth_angle_3d(
        launch['x'], launch['y'],
        target['x'], target['y']
    )
    print(f"方位角（水平方向）: {azimuth:.2f}° （y軸正方向が0°、右: +, 左: -）")
    
    horizontal_dist = calc_horizontal_distance(
        launch['x'], launch['y'],
        target['x'], target['y']
    )
    print(f"水平距離: {horizontal_dist:.2f} m")
    print(f"高さ差: {target['z'] - launch['z']:.2f} m")
    
    angles = calc_angle_for_target_2d(
        v0=v0,
        z0=launch['z'],
        horizontal_dist=horizontal_dist,
        z_target=target['z']
    )
    
    if angles is None:
        v_min = calc_min_velocity_3d(
            z0=launch['z'],
            horizontal_dist=horizontal_dist,
            z_target=target['z']
        )
        print(f"到達不可能: 速度 {v0:.2f} m/s では届きません")
        print(f"必要最小速度: {v_min:.2f} m/s")
        
        result = {
            'launch_pos': launch_pos,
            'target_pos': target_pos,
            'launch': launch,
            'target': target,
            'v0': v0,
            'azimuth': azimuth,
            'horizontal_distance': horizontal_dist,
            'height_diff': target['z'] - launch['z'],
            'reachable': False,
            'min_velocity': v_min
        }
        return result
    
    print(f"到達可能な仰角: {[f'{a:.2f}°' for a in angles]}")
    
    selected_angle = angles[0]
    print(f"使用する仰角: {selected_angle:.2f}°")
    print(f"発射速度: {v0:.2f} m/s")
    result = {
        'launch_pos': launch_pos,
        'target_pos': target_pos,
        'launch': launch,
        'target': target,
        'v0': v0,
        'azimuth': azimuth,
        'elevation': selected_angle,
        'horizontal_distance': horizontal_dist,
        'height_diff': target['z'] - launch['z'],
        'reachable': True
    }
    
    return result

if __name__ == "__main__":
    result1 = fire(1, 7)
    
    result2 = fire(0, 2)
    
    result3 = fire(2, 6)