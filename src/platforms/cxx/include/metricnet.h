#include "featurenet.h"

using namespace std;

struct PredictResultMetricNet {
    float results[1];
};

const int BATCH_SIZE=1;

class TFLiteMetricNet {
public:
    TFLiteMetricNet();
    ~TFLiteMetricNet();
    void predict(float feat1[EMBEDDING_SIZE], float feat2[BATCH_SIZE*EMBEDDING_SIZE], PredictResultMetricNet *res);
    // Methods
    void initModel(const char *model, long modelSize);
    void initModel(const char *model);
    void initModel(const char *model, int numThreads);
private:
    // members
    char *m_modelBytes = nullptr;
    unique_ptr<tflite::FlatBufferModel> m_model;
    unique_ptr<tflite::Interpreter> m_interpreter;

};
