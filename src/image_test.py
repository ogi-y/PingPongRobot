import cv2
import os


# 画像を読み込む
img_path = os.path.join(os.path.dirname(__file__), '../data/pic/pic.jpg')
image = cv2.imread(img_path)
# サンプル画像表示
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()