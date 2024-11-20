#ifndef SPOOFING_H
#define SPOOFING_H

#include "pipeline.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class Spoofing : public Pipeline {
public:
    Spoofing(const std::string& model_path);
    std::vector<float> preprocess(const std::vector<float>& input) override;
    std::vector<float> postprocess(const std::vector<float>& input) override;

private:
    cv::Mat preprocessImage(const std::string& image_path);
};

#endif // SPOOFING_H