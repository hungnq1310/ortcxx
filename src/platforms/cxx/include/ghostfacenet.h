#ifndef GHOSTFACENET_H
#define GHOSTFACENET_H

#include <opencv2/opencv.hpp>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class GhostFaceNet {
public:
    GhostFaceNet(const std::string& model_path);
    std::vector<cv::Mat> preprocess(const cv::Mat& image, const std::vector<cv::Rect>& xyxys, const std::vector<std::vector<float>>& kpts);
    std::vector<float> inference(const cv::Mat& image, const std::vector<cv::Rect>& xyxys, const std::vector<std::vector<float>>& kpts, bool norm = false);
    void postprocess(const cv::Mat& image);

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    std::string input_name;
    std::string output_name;
    cv::Size model_input_size;

    Ort::Session load_model(const std::string& path);
};

#endif // GHOSTFACENET_H