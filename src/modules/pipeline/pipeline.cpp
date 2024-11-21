#include "pipeline.h"
#include <iostream>
#include <stdexcept>

Pipeline::Pipeline(Model model)
    : env(ORT_LOGGING_LEVEL_WARNING, "Pipeline"),
      session_options(),
      session(env, model->GetModelPath().c_str(), session_options) {
    // Additional initialization if needed
}

Ort::Value Pipeline::preprocess(Ort::Value input) {
    // Default implementation: return the input as is
    std::cout << "Preprocessing..." << std::endl;
    return input;
}

Ort::Value Pipeline::postprocess(Ort::Value input) {
    // Default implementation: return the input as is
    std::cout << "Postprocessing..." << std::endl;
    return input;
}

Ort::Value Pipeline::inference(Ort::Value input) {
    // Raise an error if this method is not implemented in a derived class
    throw std::runtime_error("Inference method not implemented");
}