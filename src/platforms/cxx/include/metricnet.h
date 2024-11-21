// #include "featurenet.h"

// using namespace std;

// struct PredictResultMetricNet {
//     float results[1];
// };

// const int BATCH_SIZE=1;

// class TFLiteMetricNet {
// public:
//     TFLiteMetricNet();
//     ~TFLiteMetricNet();
//     void predict(float feat1[EMBEDDING_SIZE], float feat2[BATCH_SIZE*EMBEDDING_SIZE], PredictResultMetricNet *res);
//     // Methods
//     void initModel(const char *model, long modelSize);
//     void initModel(const char *model);
//     void initModel(const char *model, int numThreads);
// private:
//     // members
//     char *m_modelBytes = nullptr;
//     unique_ptr<tflite::FlatBufferModel> m_model;
//     unique_ptr<tflite::Interpreter> m_interpreter;

// };

#ifndef METRICNET_H
#define METRICNET_H

#include "pipeline.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <memory>
#include <string>

struct PredictResultMetricNet {
    float results[1];
};

const int BATCH_SIZE = 1;
const int EMBEDDING_SIZE = 512; // Assuming EMBEDDING_SIZE is 512

class MetricNet : public Pipeline {
public:
    MetricNet(const std::string& model_path);
    std::vector<float> preprocess(const std::vector<float>& input) override;
    std::vector<float> postprocess(const std::vector<float>& input) override;
    void predict(float feat1[EMBEDDING_SIZE], float feat2[BATCH_SIZE * EMBEDDING_SIZE], PredictResultMetricNet *res);

private:
    void initModel(const std::string& model_path);
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;
};

#endif // METRICNET_H