#include "pipeline.h"
#include <iostream>

Pipeline::Pipeline(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "Pipeline"),
      session_options(),
      session(env, model_path.c_str(), session_options) {
    // Additional initialization if needed
}

std::vector<float> Pipeline::preprocess(const std::vector<float>& input) {
    // Default implementation: return the input as is
    std::cout << "Preprocessing..." << std::endl;
    return input;
}

std::vector<float> Pipeline::postprocess(const std::vector<float>& input) {
    // Default implementation: return the input as is
    std::cout << "Postprocessing..." << std::endl;
    return input;
}