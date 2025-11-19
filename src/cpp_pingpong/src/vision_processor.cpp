#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <vector>
#include <string>
#include <sstream>
#include <iostream>

using namespace std::chrono_literals;

// 年齢クラスと、JSON出力用の中央値(int)のマッピング
const std::vector<std::string> AGE_LIST_STR = {
    "0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"
};
const std::vector<int> AGE_LIST_INT = {
    1, 5, 10, 17, 28, 40, 50, 80
};

const cv::Scalar MODEL_MEAN_VALUES = cv::Scalar(78.4263377603, 87.7689143744, 114.895847746);

class VisionProcessor : public rclcpp::Node {
public:
    VisionProcessor() : Node("vision_processor") {
        // サブスクライバー
        sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&VisionProcessor::image_callback, this, std::placeholders::_1));

        sub_age_trigger_ = this->create_subscription<std_msgs::msg::Bool>(
            "/age_trigger", 10,
            std::bind(&VisionProcessor::age_trigger_callback, this, std::placeholders::_1));

        sub_body_trigger_ = this->create_subscription<std_msgs::msg::Bool>(
            "/body_trigger", 10,
            std::bind(&VisionProcessor::body_trigger_callback, this, std::placeholders::_1));

        // パブリッシャー
        pub_age_ = this->create_publisher<std_msgs::msg::String>("/age", 10);
        pub_player_pos_ = this->create_publisher<std_msgs::msg::String>("/player_pos", 10);

        // パラメータ宣言
        this->declare_parameter("left_threshold", 0.33);
        this->declare_parameter("right_threshold", 0.66);

        // モデルの読み込み
        load_models();

        RCLCPP_INFO(this->get_logger(), "Vision Processor Node Initialized (C++)");
    }

private:
    // メンバ変数
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_age_trigger_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_body_trigger_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_age_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_player_pos_;

    cv::dnn::Net faceNet_;
    cv::dnn::Net ageNet_;
    cv::dnn::Net poseNet_;

    cv::Mat latest_frame_;
    bool prev_age_trigger_ = false;
    bool prev_body_trigger_ = false;

    // モデルロード処理
    void load_models() {
        try {
            // ファイルパスは適宜絶対パスやパラメータに変更してください
            faceNet_ = cv::dnn::readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel");
            ageNet_ = cv::dnn::readNet("age_deploy.prototxt", "age_net.caffemodel");
            poseNet_ = cv::dnn::readNetFromONNX("yolov8n-pose.onnx");

            // CPU設定
            faceNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            faceNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            ageNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            ageNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            poseNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            poseNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        } catch (const cv::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Model loading failed: %s", e.what());
        }
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // ROS画像 -> OpenCV画像変換
            latest_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void age_trigger_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (prev_age_trigger_ != msg->data) {
            RCLCPP_INFO(this->get_logger(), "Age trigger changed: %s", msg->data ? "true" : "false");
            prev_age_trigger_ = msg->data;
        }

        if (!msg->data) return;
        if (latest_frame_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No image frame available for age estimation");
            return;
        }

        // --- 顔検出 ---
        cv::Mat frame = latest_frame_.clone(); // 推論用にクローン
        int frameHeight = frame.rows;
        int frameWidth = frame.cols;

        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0));
        faceNet_.setInput(blob);
        cv::Mat detection = faceNet_.forward();
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        // JSON構築用ストリーム (簡易実装)
        std::stringstream json_ss;
        json_ss << "{\"ages\": [";
        bool first_entry = true;
        int id_counter = 1;

        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            if (confidence > 0.7) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

                x1 = std::max(0, x1); y1 = std::max(0, y1);
                x2 = std::min(frameWidth - 1, x2); y2 = std::min(frameHeight - 1, y2);

                cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
                if (roi.width < 10 || roi.height < 10) continue;

                // --- 年齢推定 ---
                cv::Mat face = frame(roi);
                cv::Mat ageBlob = cv::dnn::blobFromImage(face, 1.0, cv::Size(227, 227), MODEL_MEAN_VALUES, false);
                ageNet_.setInput(ageBlob);
                cv::Mat agePreds = ageNet_.forward();

                cv::Point maxLoc;
                cv::minMaxLoc(agePreds, 0, 0, 0, &maxLoc);
                
                // 年齢リストのインデックスから整数値(近似値)を取得
                int estimated_age = AGE_LIST_INT[maxLoc.x];

                if (!first_entry) json_ss << ", ";
                json_ss << "{\"id\": " << id_counter++ << ", \"age\": " << estimated_age << "}";
                first_entry = false;
            }
        }
        json_ss << "]}";

        // メッセージ送信
        std_msgs::msg::String out_msg;
        out_msg.data = json_ss.str();
        pub_age_->publish(out_msg);
    }

    void body_trigger_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        // 1. トリガーチェック
        if (prev_body_trigger_ != msg->data) {
            RCLCPP_INFO(this->get_logger(), "Body trigger changed: %s", msg->data ? "true" : "false");
            prev_body_trigger_ = msg->data;
        }
        if (!msg->data) return;
        if (latest_frame_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No image frame available for body detection");
            return;
        }

        // 2. エラーハンドリングを追加
        try {
            cv::Mat frame = latest_frame_.clone();
            
            // YOLOv8 Pose 推論
            cv::Mat blob;
            // 入力サイズは変換時と同じ 640x640 にする
            cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
            poseNet_.setInput(blob);

            // 修正点: 具体的な出力レイヤー名 "output0" を指定して取得する
            // これによりOpenCVが余計なレイヤー計算をするのを防ぎます
            cv::Mat out = poseNet_.forward("output0"); 

            // 次元情報の取得
            // 固定サイズで出力した場合、YOLOv8は通常 [1, 56, 8400] を返します
            int dim_channel = 1; // 56
            int dim_anchor = 2;  // 8400

            // 万が一 [1, 8400, 56] で来た場合のガード
            if (out.dims > 2 && out.size[1] > out.size[2]) {
                 dim_channel = 2; 
                 dim_anchor = 1;
            }

            int num_anchors = out.size[dim_anchor];
            int stride = num_anchors;

            // 生データへのポインタ
            float* data = (float*)out.data;

            float x_factor = (float)frame.cols / 640.0;
            float y_factor = (float)frame.rows / 640.0;

            float max_conf = -1.0;
            cv::Point2f best_nose_pos;
            bool body_detected = false;

            for (int i = 0; i < num_anchors; ++i) {
                // クラス信頼度 (Index 4)
                float confidence = data[4 * stride + i];

                if (confidence > 0.5) {
                    if (confidence > max_conf) {
                        max_conf = confidence;

                        // Keypoint 0 (Nose) [x, y] = Index 5, 6
                        float nose_x = data[5 * stride + i] * x_factor;
                        float nose_y = data[6 * stride + i] * y_factor;

                        best_nose_pos = cv::Point2f(nose_x, nose_y);
                        body_detected = true;
                    }
                }
            }

            if (!body_detected) {
                RCLCPP_INFO(this->get_logger(), "No bodies detected");
                return;
            }

            // --- 位置判定 ---
            int w = frame.cols;
            double left_thresh = this->get_parameter("left_threshold").as_double();
            double right_thresh = this->get_parameter("right_threshold").as_double();
            int position = 1; // center

            if (best_nose_pos.x < w * left_thresh) {
                position = 0; // left
            } else if (best_nose_pos.x > w * right_thresh) {
                position = 2; // right
            }

            // JSON構築
            std::stringstream json_ss;
            json_ss << "{\"pos\": " << position 
                    << ", \"x\": " << static_cast<int>(best_nose_pos.x) 
                    << ", \"y\": " << static_cast<int>(best_nose_pos.y) << "}";

            std_msgs::msg::String out_msg;
            out_msg.data = json_ss.str();
            pub_player_pos_->publish(out_msg);

        } catch (const cv::Exception& e) {
            // OpenCV内部エラーをキャッチしてノードの停止を防ぐ
            RCLCPP_ERROR(this->get_logger(), "OpenCV Error during pose estimation: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Standard Error: %s", e.what());
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VisionProcessor>();
    rclcpp::spin(node); // MultiThreadedExecutorが必要な場合は適宜変更
    rclcpp::shutdown();
    return 0;
}