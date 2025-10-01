import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace

class PersonAgeGenderDetector:
    def __init__(self):
        # MTCNNで顔検出
        self.detector = MTCNN()
        
    def detect_faces(self, frame):
        """顔を検出"""
        # BGRからRGBに変換（MTCNNはRGBフォーマットを期待）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = self.detector.detect_faces(rgb_frame)
        return face_locations
    
    def analyze_face(self, frame, face_location):
        """年齢、性別、感情を分析"""
        # BGRからRGBに変換
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        x, y, w, h = face_location['box']
        detected_face = rgb_frame[y:y+h, x:x+w]
        
        if detected_face.size == 0:
            return None
            
        try:
            result = DeepFace.analyze(detected_face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            return result
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
    def get_person_coordinates(self, face_location, frame_shape):
        """人の座標を算出（画像中心からの相対位置）"""
        h, w = frame_shape[:2]
        x, y, width, height = face_location['box']
        
        # 顔の中心座標
        center_x = x + width // 2
        center_y = y + height // 2
        
        # 画像中心からの相対位置（正規化）
        relative_x = (center_x - w/2) / (w/2)
        relative_y = (center_y - h/2) / (h/2)
        
        return {
            'pixel_x': center_x,
            'pixel_y': center_y,
            'relative_x': relative_x,
            'relative_y': relative_y,
            'bbox': (x, y, x + width, y + height)
        }
    
    def process_frame(self, frame):
        """1フレームを処理"""
        face_locations = self.detect_faces(frame)
        results = []
        
        print(f"Found {len(face_locations)} face(s) in the frame.")
        
        for index, face_location in enumerate(face_locations):
            print(f"\nFace {index+1}: {face_location['box']}")
            
            analysis_result = self.analyze_face(frame, face_location)
            coords = self.get_person_coordinates(face_location, frame.shape)
            
            if analysis_result:
                # DeepFaceの結果がリストの場合は最初の要素を取得
                if isinstance(analysis_result, list):
                    analysis_result = analysis_result[0]
                
                age = analysis_result['age']
                gender = analysis_result['dominant_gender']
                emotion = analysis_result['dominant_emotion']
                
                print(f"Age: {age}")
                print(f"Gender: {gender}")
                print(f"Emotion: {emotion}")
                
                results.append({
                    'age': age,
                    'gender': gender,
                    'emotion': emotion,
                    'coordinates': coords,
                    'confidence': face_location['confidence']
                })
            else:
                results.append({
                    'age': 'Unknown',
                    'gender': 'Unknown',
                    'emotion': 'Unknown',
                    'coordinates': coords,
                    'confidence': face_location['confidence']
                })
        
        return results


# 使用例
def main():
    detector = PersonAgeGenderDetector()
    cap = cv2.VideoCapture(0)  # Webカメラ
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.process_frame(frame)
        
        # 結果を描画
        for result in results:
            coords = result['coordinates']
            x1, y1, x2, y2 = coords['bbox']
            
            # 矩形と情報を描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{result['gender']}, Age:{result['age']}, {result['emotion']}"
            cv2.putText(frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 座標情報を表示
            coord_text = f"X:{coords['relative_x']:.2f}, Y:{coords['relative_y']:.2f}"
            cv2.putText(frame, coord_text, (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        cv2.imshow('Person Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 静的画像用の関数
def analyze_image(image_path):
    """静的画像を分析する関数"""
    detector = PersonAgeGenderDetector()
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    results = detector.process_frame(image)
    
    # 結果を描画
    for result in results:
        coords = result['coordinates']
        x1, y1, x2, y2 = coords['bbox']
        
        # 矩形と情報を描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{result['gender']}, Age:{result['age']}, {result['emotion']}"
        cv2.putText(image, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 結果を表示
    cv2.imshow('Analysis Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Webカメラでリアルタイム検出
    main()
    
    # 静的画像の分析例（必要に応じてコメントアウトを外してください）
    # analyze_image('path_to_your_image.jpg')