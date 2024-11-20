#ifndef YOLOV7_H
#define YOLOV7_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class YoloV7 : public Pipeline {
public:
    YoloV7(const std::string& model_path);
    std::vector<float> preprocess(const std::vector<float>& input) override;
    std::vector<float> postprocess(const std::vector<float>& input) override;

private:
    cv::Mat preprocessImage(const std::string& image_path);
    std::vector<float> postprocessDetections(const cv::Mat& detections);
};

#endif // YOLOV7_H