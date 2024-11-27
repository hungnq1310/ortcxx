#ifndef __ORTCXX_PIPELINE_H__
#define __ORTCXX_PIPELINE_H__

#include <string>
#include <ortcxx/model.h>
#include <onnxruntime_cxx_api.h>

using namespace ortcxx::model;

class Pipeline {
    public:
        Pipeline(Model model);
        virtual ~Pipeline() = default;

        virtual Ort::Value preprocess(Ort::Value input);
        virtual Ort::Value postprocess(Ort::Value input);
        virtual Ort::Value inference(Ort::Value input);

    protected:
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session session;
    };

#endif // __ORTCXX_PIPELINE_H__