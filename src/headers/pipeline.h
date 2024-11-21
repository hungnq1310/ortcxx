#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include <model.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

class Pipeline {
    public:
        Pipeline(Model model);
        virtual ~Pipeline() = default;

        virtual Ort::Value preprocess(Ort::Value input);
        virtual Ort::Value postprocess(Ort::Value input);

    protected:
        Ort::Env env;
        Ort::SessionOptions session_options;
        Ort::Session session;
    };

#endif // PIPELINE_H