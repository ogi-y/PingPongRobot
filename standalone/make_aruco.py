import cv2
import numpy as np

def generate_aruco_grid(id_list, marker_size=200, dict_type=cv2.aruco.DICT_4X4_50, grid_shape=(3, 3), margin=20, out_path="aruco_grid.png"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
    rows, cols = grid_shape
    assert len(id_list) <= rows * cols, "id_listの数がグリッドサイズを超えています"

    # A4サイズ（300dpi）
    a4_width_px = 2480  # 210mm * 300 / 25.4
    a4_height_px = 3508 # 297mm * 300 / 25.4
    canvas = 255 * np.ones((a4_height_px, a4_width_px), dtype=np.uint8)

    # グリッドの最大サイズをA4内に収める（外周マージンも含めて計算）
    rows, cols = grid_shape
    n_h_margin = rows + 1  # 横方向の隙間数
    n_v_margin = cols + 1  # 縦方向の隙間数
    # 外周マージンを内側マージンの半分とする
    # まず仮でmargin=marker_size//4で計算し、A4に収まるようにスケーリング
    margin_ratio = 0.25
    # まず最大マーカーサイズを計算
    max_marker_w = a4_width_px / (cols + (cols + 1) * margin_ratio)
    max_marker_h = a4_height_px / (rows + (rows + 1) * margin_ratio)
    marker_size = int(min(max_marker_w, max_marker_h))
    margin = int(marker_size * margin_ratio)
    outer_margin = margin // 2
    grid_width = cols * marker_size + (cols - 1) * margin + 2 * outer_margin
    grid_height = rows * marker_size + (rows - 1) * margin + 2 * outer_margin
    grid_img = 255 * np.ones((grid_height, grid_width), dtype=np.uint8)

    for idx, marker_id in enumerate(id_list):
        r = idx // cols
        c = idx % cols
        marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img, 1)
        y = outer_margin + r * (marker_size + margin)
        x = outer_margin + c * (marker_size + margin)
        grid_img[y:y+marker_size, x:x+marker_size] = marker_img

    # カットライン（黒線）を隙間中央に描画
    line_thickness = max(2, marker_size // 60)
    for r in range(1, rows):
        y_line = outer_margin + r * marker_size + (r - 0.5) * margin
        y_line = int(round(y_line))
        grid_img[y_line-line_thickness//2:y_line+line_thickness//2+1, :] = 0
    for c in range(1, cols):
        x_line = outer_margin + c * marker_size + (c - 0.5) * margin
        x_line = int(round(x_line))
        grid_img[:, x_line-line_thickness//2:x_line+line_thickness//2+1] = 0

    # グリッドをA4中央に貼り付け
    y_offset = (a4_height_px - grid_height) // 2
    x_offset = (a4_width_px - grid_width) // 2
    canvas[y_offset:y_offset+grid_height, x_offset:x_offset+grid_width] = grid_img

    cv2.imwrite(out_path, canvas)
    print(f"Saved: {out_path}")

    # カットライン（黒線）を隙間中央に描画
    line_thickness = 3
    # 横方向
    for r in range(1, rows):
        # 各マーカーの下端からmargin/2だけ下がった位置
        y_line = outer_margin + r * marker_size + (r - 0.5) * margin
        y_line = int(round(y_line))
        grid_img[y_line-line_thickness//2:y_line+line_thickness//2+1, :] = 0
    # 縦方向
    for c in range(1, cols):
        x_line = outer_margin + c * marker_size + (c - 0.5) * margin
        x_line = int(round(x_line))
        grid_img[:, x_line-line_thickness//2:x_line+line_thickness//2+1] = 0

    cv2.imwrite(out_path, grid_img)
    print(f"Saved: {out_path}")

generate_aruco_grid(list(range(0, 9)), marker_size=300, grid_shape=(3, 3), out_path="aruco_grid_0_8.png")