#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// 年齢のクラス定義（Levi and Hassnerモデルの仕様）
const vector<string> AGE_LIST = {
    "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"
};

// モデルの平均値（正規化用）
const Scalar MODEL_MEAN_VALUES = Scalar(78.4263377603, 87.7689143744, 114.895847746);

int main() {
    // --------------------------------------------------
    // 1. モデルの読み込み
    // --------------------------------------------------
    string faceProto = "deploy.prototxt";
    string faceModel = "res10_300x300_ssd_iter_140000.caffemodel";
    string ageProto = "age_deploy.prototxt";
    string ageModel = "age_net.caffemodel";

    Net faceNet, ageNet;
    try {
        faceNet = readNet(faceProto, faceModel);
        ageNet = readNet(ageProto, ageModel);
        cout << "モデルの読み込み完了。" << endl;
    } catch (const cv::Exception& e) {
        cerr << "モデルファイルの読み込みに失敗しました。パスを確認してください。" << endl;
        return -1;
    }

    // 高速化: GPUが使えるならバックエンドをCUDAにする（なければCPUで動作します）
    faceNet.setPreferableBackend(DNN_BACKEND_DEFAULT);
    faceNet.setPreferableTarget(DNN_TARGET_CPU);
    ageNet.setPreferableBackend(DNN_BACKEND_DEFAULT);
    ageNet.setPreferableTarget(DNN_TARGET_CPU);

    // --------------------------------------------------
    // 2. Webカメラの起動
    // --------------------------------------------------
    VideoCapture cap(0); // 0番目のカメラを開く
    if (!cap.isOpened()) {
        cerr << "Webカメラが開けませんでした。" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 画像のリサイズ（処理高速化のため、幅を小さくしてアスペクト比維持）
        // 必要なければコメントアウトしてください
        resize(frame, frame, Size(640, 480));

        int frameHeight = frame.rows;
        int frameWidth = frame.cols;

        // --------------------------------------------------
        // 3. 顔検出の実行
        // --------------------------------------------------
        // 顔検出モデルへの入力作成 (300x300にリサイズ)
        Mat blob = blobFromImage(frame, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0));
        faceNet.setInput(blob);
        Mat detection = faceNet.forward();

        // 検出結果の行列を取得
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            // 信頼度が0.7以上のものだけを処理
            if (confidence > 0.7) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

                // 画面外にはみ出さないようにクリップ
                x1 = max(0, x1); y1 = max(0, y1);
                x2 = min(frameWidth - 1, x2); y2 = min(frameHeight - 1, y2);

                // 顔領域（ROI）の切り出し
                Rect roi(x1, y1, x2 - x1, y2 - y1);
                // 矩形が小さすぎる場合はスキップ
                if (roi.width < 10 || roi.height < 10) continue; 
                Mat face = frame(roi);

                // --------------------------------------------------
                // 4. 年齢推定の実行
                // --------------------------------------------------
                // 年齢推定モデルへの入力作成 (227x227にリサイズ、平均値を引く)
                Mat ageBlob = blobFromImage(face, 1.0, Size(227, 227), MODEL_MEAN_VALUES, false);
                ageNet.setInput(ageBlob);
                Mat agePreds = ageNet.forward();

                // 最も確率の高いクラスを探す
                Point maxLoc;
                double maxVal;
                minMaxLoc(agePreds, 0, &maxVal, 0, &maxLoc);
                
                // 結果の文字列
                string label = AGE_LIST[maxLoc.x];
                
                // --------------------------------------------------
                // 5. 結果の描画
                // --------------------------------------------------
                rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
                putText(frame, label, Point(x1, y1 - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
            }
        }

        imshow("Age Estimation Demo", frame);

        // 'q'キーで終了
        if (waitKey(1) == 'q') break;
    }

    return 0;
}