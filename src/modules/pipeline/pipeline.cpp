#include <iostream>
#include <stdexcept>
#include <ortcxx/pipeline.h>

using namespace ortcxx::model;
using namespace ortcxx::pipeline;

Pipeline::Pipeline(Model* model){
    this->model = model;
}

Pipeline::~Pipeline() {
    // Release the session
    // this->session = nullptr;
    this->model = nullptr;
}

Ort::Value* Pipeline::preprocess(Ort::Value* input) {
    // Default implementation: return the input as is
    std::cout << "Preprocessing..." << std::endl;
    return input;
}

Ort::Value* Pipeline::postprocess(Ort::Value* input) {
    // Default implementation: return the input as is
    std::cout << "Postprocessing..." << std::endl;
    return input;
}

Ort::Value* Pipeline::inference(Ort::Value* input) {
    // Raise an error if this method is not implemented in a derived class
    throw std::runtime_error("Inference method not implemented");
}