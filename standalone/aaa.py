import time
from deepface import DeepFace
import cv2

# 推定したい属性をここで指定
# actions = ['age', 'gender', 'emotion', 'race']
actions = ['age']

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    
    # FPS計測
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time

    try:
        result = DeepFace.analyze(frame, actions=actions, enforce_detection=False)
        result = result[0]  # DeepFaceの戻り値はリスト

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

        region = result.get('region', None)
        if region:
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # 推定結果の表示
        cv2.putText(frame, " ".join(display_text), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPSの表示
        cv2.putText(frame, f"FPS: {fps:.2f}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    except Exception as e:
        pass
    
    cv2.imshow('Age Estimation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()