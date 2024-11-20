#include "spoofing.h"
#include <iostream>

Spoofing::Spoofing(const std::string& model_path)
    : Pipeline(model_path) {
    // Additional initialization if needed
}

std::vector<float> Spoofing::preprocess(const std::vector<float>& input) {
    // Implement preprocessing logic here
    std::cout << "Spoofing Preprocessing..." << std::endl;
    // Example: convert input to cv::Mat and preprocess
    std::string image_path = "path/to/your/image.jpg"; // Replace with actual path
    cv::Mat preprocessed_image = preprocessImage(image_path);

    // Convert cv::Mat to std::vector<float>
    std::vector<float> output(preprocessed_image.begin<float>(), preprocessed_image.end<float>());
    return output;
}

std::vector<float> Spoofing::postprocess(const std::vector<float>& input) {
    // Implement postprocessing logic here
    std::cout << "Spoofing Postprocessing..." << std::endl;
    // Example: return the input as is
    return input;
}

cv::Mat Spoofing::preprocessImage(const std::string& image_path) {
    // Load the image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not open or find the image!" << std::endl;
        return cv::Mat();
    }

    // Convert the image to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Resize the image to (256, 256)
    cv::resize(img, img, cv::Size(256, 256));

    // Convert the image to float and normalize to [0, 1]
    img.convertTo(img, CV_32F, 1.0 / 255);

    // Normalize the image
    cv::Mat mean = cv::Mat(img.size(), img.type(), cv::Scalar(0.5, 0.5, 0.5));
    cv::Mat std = cv::Mat(img.size(), img.type(), cv::Scalar(0.5, 0.5, 0.5));
    cv::subtract(img, mean, img);
    cv::divide(img, std, img);

    return img;
}