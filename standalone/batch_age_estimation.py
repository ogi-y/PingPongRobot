import os
import cv2
from deepface import DeepFace
from pathlib import Path

# 設定
INPUT_FOLDER = "./standalone/pic"  # 入力フォルダのパス
OUTPUT_FOLDER = "./standalone/pic/output"  # 出力フォルダのパス
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# 推定したい属性
actions = ['age']

def process_image(image_path, output_path):
    """
    画像を読み込んで年齢推定を行い、結果をオーバーレイした画像を保存する
    
    Args:
        image_path: 入力画像のパス
        output_path: 出力画像のパス
    """
    print(f"Processing: {image_path}")
    
    # 画像を読み込む
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"  Error: Could not read image {image_path}")
        return False
    
    try:
        # DeepFaceで年齢推定
        results = DeepFace.analyze(frame, actions=actions, enforce_detection=False)
        
        # 複数の顔が検出された場合に対応
        if not isinstance(results, list):
            results = [results]
        
        # 検出された各顔に対して処理
        for i, result in enumerate(results):
            # 表示するテキストを構築
            display_text = []
            if 'age' in actions:
                display_text.append(f"Age: {int(result['age'])}")
            if 'gender' in actions:
                gender = result.get('dominant_gender', result.get('gender', ''))
                display_text.append(f"Gender: {gender}")
            if 'emotion' in actions:
                emotion = result.get('dominant_emotion', result.get('emotion', ''))
                display_text.append(f"Emotion: {emotion}")
            if 'race' in actions:
                race = result.get('dominant_race', result.get('race', ''))
                display_text.append(f"Race: {race}")
            
            # 顔の領域を取得
            region = result.get('region', None)
            if region:
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # 顔の周りに矩形を描画
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # テキスト設定
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                line_height = 25
                
                # 各行のテキストを描画
                for idx, text in enumerate(display_text):
                    # テキストのサイズを取得
                    (text_width, text_height), baseline = cv2.getTextSize(
                        text, font, font_scale, thickness
                    )
                    
                    # テキストの位置を計算（顔の上部に複数行表示）
                    text_y = y - (len(display_text) - idx) * line_height
                    
                    # テキストの背景矩形を描画
                    cv2.rectangle(
                        frame,
                        (x, text_y - text_height - 5),
                        (x + text_width + 10, text_y + 5),
                        (255, 0, 0),
                        -1  # 塗りつぶし
                    )
                    
                    # テキストを描画
                    cv2.putText(
                        frame,
                        text,
                        (x + 5, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )
        
        # 処理された画像を保存
        cv2.imwrite(str(output_path), frame)
        print(f"  Saved: {output_path} ({len(results)} face(s) detected)")
        return True
        
    except Exception as e:
        print(f"  Error processing {image_path}: {str(e)}")
        # エラーが発生しても元画像を保存
        cv2.imwrite(str(output_path), frame)
        return False

def main():
    """
    メイン処理: フォルダ内の全画像を処理
    """
    # 入力フォルダの確認
    input_path = Path(INPUT_FOLDER)
    if not input_path.exists():
        print(f"Error: Input folder does not exist: {INPUT_FOLDER}")
        return
    
    # 出力フォルダの作成
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # 画像ファイルを取得
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        return
    
    print(f"Found {len(image_files)} image(s)")
    print("-" * 50)
    
    # 各画像を処理
    success_count = 0
    for image_file in image_files:
        output_file = output_path / f"age_{image_file.name}"
        if process_image(image_file, output_file):
            success_count += 1
        print()
    
    print("-" * 50)
    print(f"Completed: {success_count}/{len(image_files)} images processed successfully")

if __name__ == "__main__":
    main()
