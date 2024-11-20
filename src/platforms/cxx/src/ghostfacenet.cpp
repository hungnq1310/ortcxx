#include "ghostfacenet.h"
#include "face.h" // Assuming face.h contains the align_5_points function
#include "crop_image.h" // Assuming crop_image.h contains the crop_image function

GhostFaceNet::GhostFaceNet(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "GhostFaceNet"),
      session_options(),
      session(load_model(model_path)) {
    input_name = session.GetInputName(0, env);
    output_name = session.GetOutputName(0, env);
    auto input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    model_input_size = cv::Size(input_shape[2], input_shape[1]);
}

Ort::Session GhostFaceNet::load_model(const std::string& path) {
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return Ort::Session(env, path.c_str(), session_options);
}

std::vector<cv::Mat> GhostFaceNet::preprocess(const cv::Mat& image, const std::vector<cv::Rect>& xyxys, const std::vector<std::vector<float>>& kpts) {
    std::vector<cv::Mat> crops;
    for (size_t i = 0; i < xyxys.size(); ++i) {
        const auto& box = xyxys[i];
        const auto& kpt = kpts[i];
        cv::Mat crop = crop_image(image, box);
        std::vector<float> aligned_kpt = kpt;
        for (size_t j = 0; j < kpt.size(); j += 3) {
            aligned_kpt[j] -= box.x;
            aligned_kpt[j + 1] -= box.y;
        }
        crop = face::align_5_points(crop, aligned_kpt);
        cv::resize(crop, crop, model_input_size);
        crop.convertTo(crop, CV_32F, 1.0 / 255);
        crop = (crop - 0.5) * 2.0;
        crops.push_back(crop);
    }
    return crops;
}

std::vector<float> GhostFaceNet::inference(const cv::Mat& image, const std::vector<cv::Rect>& xyxys, const std::vector<std::vector<float>>& kpts, bool norm) {
    std::vector<cv::Mat> crops = preprocess(image, xyxys, kpts);
    std::vector<float> input_tensor_values;
    for (const auto& crop : crops) {
        input_tensor_values.insert(input_tensor_values.end(), crop.begin<float>(), crop.end<float>());
    }

    std::vector<int64_t> input_shape = {static_cast<int64_t>(crops.size()), model_input_size.height, model_input_size.width, 3};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(env, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    std::vector<float> result = output_tensors.front().GetTensorMutableData<float>();

    if (norm) {
        for (size_t i = 0; i < result.size(); i += 512) { // Assuming embedding size is 512
            float norm_factor = 0.0;
            for (size_t j = 0; j < 512; ++j) {
                norm_factor += result[i + j] * result[i + j];
            }
            norm_factor = std::sqrt(norm_factor);
            for (size_t j = 0; j < 512; ++j) {
                result[i + j] /= norm_factor;
            }
        }
    }

    return result;
}

void GhostFaceNet::postprocess(const cv::Mat& image) {
    throw std::runtime_error("Not implemented");
}