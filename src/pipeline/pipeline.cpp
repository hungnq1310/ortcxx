#include "ortcxx/pipeline.h"
#include <iostream>
#include <stdexcept>

using namespace ortcxx::model;

Pipeline::Pipeline(Model* model){
    this->model = model;
    this->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Cinnamon");
    this->session_options = Ort::SessionOptions();
    this->session = Ort::Session(
        this->env, 
        this->model->getModelPath(), 
        this->session_options
    );
}

std::shared_ptr<Ort::Value> Pipeline::preprocess(Ort::Value& input) {
    // Default implementation: return the input as is
    std::cout << "Preprocessing..." << std::endl;
    return make_shared<Ort::Value>(input);
}

std::shared_ptr<Ort::Value> Pipeline::postprocess(Ort::Value& input) {
    // Default implementation: return the input as is
    std::cout << "Postprocessing..." << std::endl;
    return make_shared<Ort::Value>(input);
}

std::shared_ptr<Ort::Value> Pipeline::inference(Ort::Value& input) {
    // Raise an error if this method is not implemented in a derived class
    throw std::runtime_error("Inference method not implemented");
}