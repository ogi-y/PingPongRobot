#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// 関節の接続関係（COCOフォーマット: 鼻-目, 目-耳, 肩-肘...）
const vector<pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
    {5, 11}, {6, 12}, {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}
};

// 関節の色（左右で色分けなど）
const Scalar COLOR_BONE = Scalar(255, 255, 0);
const Scalar COLOR_KPT = Scalar(0, 255, 0);

int main() {
    // 1. モデルのロード
    string modelPath = "yolov8n-pose.onnx";
    Net net = readNetFromONNX(modelPath);
    
    // GPU設定 (使える場合)
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 2. カメラまたは動画の読み込み
    // WSL環境でカメラ不可の場合は動画パスを指定: "dance.mp4" など
    VideoCapture cap(0); 
    if (!cap.isOpened()) {
        cerr << "カメラが開けません" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 3. 前処理 (YOLOv8は 640x640, 1/255.0, RGB)
        Mat blob;
        blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
        net.setInput(blob);

        // 4. 推論実行
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        // 5. 後処理（出力データの解析）
        // YOLOv8 Poseの出力形状: [1, 56, 8400] 
        // 56 = 4(box) + 1(conf) + 51(17個の関節 * 3要素(x,y,conf))
        Mat outputData = outputs[0]; 
        
        // 行列を転置して扱いやすくする: [1, 56, 8400] -> [8400, 56]
        // C++ OpenCVのMat操作で転置を行うためにreshapeを利用
        // ※次元の扱いがバージョンの違いで厄介なため、ポインタで直接アクセスします
        
        float* data = (float*)outputData.data;
        int dimensions = outputData.size[1]; // 56
        int rows = outputData.size[2];       // 8400

        // スケール計算（640x640から元の画像サイズに戻すため）
        float x_factor = (float)frame.cols / 640.0;
        float y_factor = (float)frame.rows / 640.0;

        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;
        vector<vector<float>> keypoints_list;

        for (int i = 0; i < rows; ++i) {
            // 各アンカー(8400個)ごとのデータの先頭ポインタ
            // dataは [channel, anchor] の順ではなく [batch, channel, anchor] なので注意が必要
            // しかしOpenCVのDNN出力は通常フラットなので、ここでは
            // [0行目: x, 1行目: y, 2: w, 3: h, 4: conf, 5~: kpts...] 
            // という「行」にデータが並んでいる形（56行 x 8400列）で返ってくることが多いです。
            
            // 修正: YOLOv8のONNX出力は [Batch, Channel, Anchors] = [1, 56, 8400]
            // これをループで回すには、メモリアクセスが飛び飛びになるため少し複雑です。
            
            float confidence = outputData.at<float>(0, 4, i);

            if (confidence >= 0.5) { // 信頼度閾値
                float x = outputData.at<float>(0, 0, i);
                float y = outputData.at<float>(0, 1, i);
                float w = outputData.at<float>(0, 2, i);
                float h = outputData.at<float>(0, 3, i);

                int left = int((x - w / 2) * x_factor);
                int top = int((y - h / 2) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
                class_ids.push_back(0); // 人クラスのみ

                // キーポイント情報の取得 (5番目以降)
                vector<float> kpts;
                for (int k = 0; k < 17; k++) {
                    // x, y, visibility
                    float kx = outputData.at<float>(0, 5 + k * 3, i) * x_factor;
                    float ky = outputData.at<float>(0, 5 + k * 3 + 1, i) * y_factor;
                    float ks = outputData.at<float>(0, 5 + k * 3 + 2, i); // score
                    kpts.push_back(kx);
                    kpts.push_back(ky);
                    kpts.push_back(ks);
                }
                keypoints_list.push_back(kpts);
            }
        }

        // NMS (Non-Maximum Suppression) で重複削除
        vector<int> indices;
        NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

        // 6. 描画
        for (int idx : indices) {
            Rect box = boxes[idx];
            rectangle(frame, box, Scalar(0, 255, 0), 2);

            vector<float> kpts = keypoints_list[idx];

            // 関節点（Keypoints）を描画
            for (int k = 0; k < 17; k++) {
                float kx = kpts[k * 3];
                float ky = kpts[k * 3 + 1];
                float ks = kpts[k * 3 + 2];

                if (ks > 0.5) { // 信頼度が高い点のみ描画
                    circle(frame, Point((int)kx, (int)ky), 4, COLOR_KPT, -1);
                }
            }

            // 骨格（Skeleton）を描画
            for (auto& bone : SKELETON) {
                int idx1 = bone.first;
                int idx2 = bone.second;

                float x1 = kpts[idx1 * 3];
                float y1 = kpts[idx1 * 3 + 1];
                float s1 = kpts[idx1 * 3 + 2];

                float x2 = kpts[idx2 * 3];
                float y2 = kpts[idx2 * 3 + 1];
                float s2 = kpts[idx2 * 3 + 2];

                if (s1 > 0.5 && s2 > 0.5) {
                    line(frame, Point((int)x1, (int)y1), Point((int)x2, (int)y2), COLOR_BONE, 2);
                }
            }
        }

        imshow("YOLOv8 Pose C++", frame);
        if (waitKey(1) == 'q') break;
    }
    return 0;
}