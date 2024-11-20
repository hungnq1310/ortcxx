#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class Pipeline {
public:
    Pipeline(const std::string& model_path);
    virtual ~Pipeline() = default;

    virtual std::vector<float> preprocess(const std::vector<float>& input);
    virtual std::vector<float> postprocess(const std::vector<float>& input);

protected:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
};

#endif // PIPELINE_H