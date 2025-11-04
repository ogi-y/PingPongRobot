from ultralytics import YOLO
import cv2

INPUT_FOLDER = "./data/images"  # 入力フォルダのパス
OUTPUT_FOLDER = "./data/images/output"  # 出力フォルダのパス


model = YOLO('yolov8n-pose.pt')
# model = YOLO('yolov8n.pt')

def process_image(image_path, output_path):
    """
    画像を読み込んで人物検出を行い、結果をオーバーレイした画像を保存する
    
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
    
    # YOLOで人物検出
    results = model(frame)
    
    # 結果を画像にオーバーレイ
    annotated_frame = results[0].plot()
    
    # 結果画像を保存
    cv2.imwrite(str(output_path), annotated_frame)
    print(f"  Saved output to: {output_path}")
    return True

if __name__ == "__main__":
    import os
    from pathlib import Path
    import glob

    input_dir = Path(INPUT_FOLDER)
    output_dir = Path(OUTPUT_FOLDER)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        image_files.extend(input_dir.glob(ext))
    image_files = sorted(image_files)

    if not image_files:
        print(f'No images found in {input_dir}. Create the directory and add some images.')
    else:
        print(f'Found {len(image_files)} images in {input_dir}')

    for img_path in image_files:
        output_path = output_dir / img_path.name
        process_image(img_path, output_path)
    
    print("Processing completed.")