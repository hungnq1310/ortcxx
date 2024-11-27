#ifndef __ORTCXX_PIPELINE_H__
#define __ORTCXX_PIPELINE_H__

#include <string>
#include <ortcxx/model.h>
#include <onnxruntime_cxx_api.h>
#include <ortcxx/pipeline.h>

using namespace ortcxx::model;
using namespace std;
namespace ortcxx::pipeline {
class Pipeline {
    public:
        Pipeline(Model* model);
        ~Pipeline();

        shared_ptr<Ort::Value> preprocess(Ort::Value& input);
        shared_ptr<Ort::Value> postprocess(Ort::Value& input);
        shared_ptr<Ort::Value> inference(Ort::Value& input);

    protected:
        Model* model;
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session session;
    };

} // namespace pipeline

#endif // __ORTCXX_PIPELINE_H__