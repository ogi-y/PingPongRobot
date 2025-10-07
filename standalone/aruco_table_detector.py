import cv2
import numpy as np
from cv2 import aruco
import json

class TableDetector:
    """ArUcoマーカーを使って卓球台エリアを検出"""
    
    def __init__(self, marker_size=0.05):
        """
        Args:
            marker_size: マーカーの実サイズ（メートル）デフォルト5cm
        """
        # ArUco辞書とパラメータの初期化
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # マーカーID と 位置の対応
        # ID 0: 左上, ID 1: 右上, ID 2: 右下, ID 3: 左下
        self.marker_positions = {
            0: 'top_left',
            1: 'top_right', 
            2: 'bottom_right',
            3: 'bottom_left'
        }
        
        self.table_corners = None
        self.table_polygon = None
        
    def detect_table(self, frame):
        """
        フレームからArUcoマーカーを検出してテーブル領域を特定
        
        Args:
            frame: 入力画像
            
        Returns:
            detected: 検出成功したか
        """
        # グレースケール変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # マーカー検出
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 4:
            return False
        
        # 4つのマーカーを検出できた場合
        detected_corners = {}
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in self.marker_positions:
                # マーカーの中心座標を計算
                corner = corners[i][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
                
                position = self.marker_positions[marker_id]
                detected_corners[position] = (center_x, center_y)
        
        # 4隅すべて検出できたか確認
        if len(detected_corners) == 4:
            self.table_corners = detected_corners
            
            # ポリゴンを作成（判定用）
            pts = np.array([
                detected_corners['top_left'],
                detected_corners['top_right'],
                detected_corners['bottom_right'],
                detected_corners['bottom_left']
            ], dtype=np.int32)
            self.table_polygon = pts
            
            return True
        
        return False
    
    def is_in_table(self, x, y):
        """
        指定座標がテーブル内か判定
        
        Args:
            x, y: 画像上の座標
            
        Returns:
            テーブル内ならTrue
        """
        if self.table_polygon is None:
            return True  # テーブル未検出時は全て有効
        
        result = cv2.pointPolygonTest(self.table_polygon, (float(x), float(y)), False)
        return result >= 0
    
    def get_zone(self, x, y):
        """
        テーブルを9分割したどのゾーンにいるか
        
        Returns:
            "top_left", "top_center", "top_right",
            "middle_left", "middle_center", "middle_right",
            "bottom_left", "bottom_center", "bottom_right"
        """
        if self.table_corners is None:
            return "unknown"
        
        # テーブル座標系に変換（0-1）
        rel_x, rel_y = self.to_table_coords(x, y)
        
        # 3分割
        if rel_x < 0.33:
            x_zone = "left"
        elif rel_x < 0.67:
            x_zone = "center"
        else:
            x_zone = "right"
        
        if rel_y < 0.33:
            y_zone = "top"
        elif rel_y < 0.67:
            y_zone = "middle"
        else:
            y_zone = "bottom"
        
        return f"{y_zone}_{x_zone}"
    
    def to_table_coords(self, x, y):
        """
        画像座標をテーブル座標系（0-1）に変換
        
        Returns:
            (rel_x, rel_y): 0.0-1.0の相対座標
        """
        if self.table_corners is None:
            return 0.5, 0.5
        
        # 簡易的な線形補間（より正確にはホモグラフィ変換を使用）
        tl = self.table_corners['top_left']
        tr = self.table_corners['top_right']
        bl = self.table_corners['bottom_left']
        br = self.table_corners['bottom_right']
        
        # X方向の相対位置（上辺と下辺の平均）
        top_width = tr[0] - tl[0]
        bottom_width = br[0] - bl[0]
        
        if abs(top_width) > 1:
            rel_x_top = (x - tl[0]) / top_width
        else:
            rel_x_top = 0.5
            
        if abs(bottom_width) > 1:
            rel_x_bottom = (x - bl[0]) / bottom_width
        else:
            rel_x_bottom = 0.5
        
        rel_x = (rel_x_top + rel_x_bottom) / 2
        
        # Y方向の相対位置（左辺と右辺の平均）
        left_height = bl[1] - tl[1]
        right_height = br[1] - tr[1]
        
        if abs(left_height) > 1:
            rel_y_left = (y - tl[1]) / left_height
        else:
            rel_y_left = 0.5
            
        if abs(right_height) > 1:
            rel_y_right = (y - tr[1]) / right_height
        else:
            rel_y_right = 0.5
        
        rel_y = (rel_y_left + rel_y_right) / 2
        
        # 0-1の範囲にクリップ
        rel_x = max(0, min(1, rel_x))
        rel_y = max(0, min(1, rel_y))
        
        return rel_x, rel_y
    
    def get_net_y(self):
        """
        ネット位置のY座標を取得（テーブル中央）
        
        Returns:
            ネットのY座標（ピクセル）
        """
        if self.table_corners is None:
            return None
        
        tl = self.table_corners['top_left']
        bl = self.table_corners['bottom_left']
        tr = self.table_corners['top_right']
        br = self.table_corners['bottom_right']
        
        # 左辺と右辺の中点の平均
        left_mid = (tl[1] + bl[1]) / 2
        right_mid = (tr[1] + br[1]) / 2
        
        return int((left_mid + right_mid) / 2)
    
    def draw_table(self, frame, color=(0, 255, 0), thickness=2):
        """
        テーブル領域を描画
        
        Args:
            frame: 描画対象のフレーム
            color: 線の色
            thickness: 線の太さ
        """
        if self.table_polygon is None:
            return frame
        
        # テーブルの輪郭を描画
        cv2.polylines(frame, [self.table_polygon], True, color, thickness)
        
        # ネットラインを描画
        net_y = self.get_net_y()
        if net_y is not None:
            tl = self.table_corners['top_left']
            tr = self.table_corners['top_right']
            bl = self.table_corners['bottom_left']
            br = self.table_corners['bottom_right']
            
            # ネット位置での左端と右端を計算
            left_x = int(tl[0] + (bl[0] - tl[0]) * 0.5)
            right_x = int(tr[0] + (br[0] - tr[0]) * 0.5)
            
            cv2.line(frame, (left_x, net_y), (right_x, net_y), (255, 255, 0), thickness)
            cv2.putText(frame, "NET", (left_x + 10, net_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 各コーナーにラベル表示
        for position, (x, y) in self.table_corners.items():
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, position.replace('_', ' ').title(), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def save_calibration(self, filename="table_calibration.json"):
        """
        テーブル座標を保存
        
        Args:
            filename: 保存ファイル名
        """
        if self.table_corners is None:
            print("No table detected to save")
            return False
        
        data = {
            'corners': {k: list(v) for k, v in self.table_corners.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Table calibration saved to {filename}")
        return True
    
    def load_calibration(self, filename="table_calibration.json"):
        """
        保存されたテーブル座標を読み込み
        
        Args:
            filename: 読み込みファイル名
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.table_corners = {k: tuple(v) for k, v in data['corners'].items()}
            
            # ポリゴンを再構築
            pts = np.array([
                self.table_corners['top_left'],
                self.table_corners['top_right'],
                self.table_corners['bottom_right'],
                self.table_corners['bottom_left']
            ], dtype=np.int32)
            self.table_polygon = pts
            
            print(f"Table calibration loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False


def generate_aruco_markers(output_dir="./aruco_markers"):
    """
    ArUcoマーカーを生成して保存
    
    Args:
        output_dir: 出力ディレクトリ
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    
    # 4つのマーカーを生成
    marker_ids = [0, 1, 2, 3]
    positions = ['TopLeft', 'TopRight', 'BottomRight', 'BottomLeft']
    
    for marker_id, position in zip(marker_ids, positions):
        # 200x200ピクセルのマーカー画像を生成
        marker_image = aruco.generateImageMarker(aruco_dict, marker_id, 200)
        
        filename = f"{output_dir}/marker_{marker_id}_{position}.png"
        cv2.imwrite(filename, marker_image)
        print(f"Generated: {filename}")
    
    print(f"\nマーカーを印刷して、机の四隅に以下の順で貼ってください:")
    print("  ID 0: 左上（Top Left）")
    print("  ID 1: 右上（Top Right）")
    print("  ID 2: 右下（Bottom Right）")
    print("  ID 3: 左下（Bottom Left）")


if __name__ == "__main__":
    # マーカー生成
    print("ArUcoマーカーを生成中...")
    generate_aruco_markers()
    
    print("\n=== テーブル検出テスト ===")
    print("カメラを起動してマーカーを検出します...")
    
    # カメラテスト
    cap = cv2.VideoCapture(0)
    detector = TableDetector()
    
    print("Press 's' to save calibration, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # テーブル検出
        detected = detector.detect_table(frame)
        
        if detected:
            # テーブル描画
            frame = detector.draw_table(frame)
            
            # ステータス表示
            cv2.putText(frame, "Table Detected!", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Searching for markers...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('ArUco Table Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and detected:
            detector.save_calibration()
    
    cap.release()
    cv2.destroyAllWindows()
