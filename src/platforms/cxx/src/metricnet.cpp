#include "metricnet.h"

TFLiteMetricNet::TFLiteMetricNet() {
}


TFLiteMetricNet::~TFLiteMetricNet() {
    if (m_modelBytes != nullptr) {
        free(m_modelBytes);
        m_modelBytes = nullptr;
    }
}


void TFLiteMetricNet::initModel(const char *tfliteModel, long modelSize) {

    // Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
    m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
    memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
    m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);
    assert(m_model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*m_model, resolver);
    builder(&m_interpreter);
    assert(m_interpreter != nullptr);

    // Allocate tensor buffers.
    assert(m_interpreter->AllocateTensors() == kTfLiteOk);
    assert(m_interpreter->Invoke() == kTfLiteOk);
}

void TFLiteMetricNet::initModel(const char *tfliteModel) {

    m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
    assert(m_model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*m_model, resolver);
    builder(&m_interpreter);
    assert(m_interpreter != nullptr);

    // Allocate tensor buffers.
    assert(m_interpreter->AllocateTensors() == kTfLiteOk);
    assert(m_interpreter->Invoke() == kTfLiteOk);
}

void TFLiteMetricNet::initModel(const char *tfliteModel, int numThreads) {

    m_model = tflite::FlatBufferModel::BuildFromFile(tfliteModel);
    assert(m_model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*m_model, resolver);
    builder(&m_interpreter);
    assert(m_interpreter != nullptr);

    // Allocate tensor buffers.
    assert(m_interpreter->AllocateTensors() == kTfLiteOk);
    assert(m_interpreter->Invoke() == kTfLiteOk);
    // interpreter->SetAllowFp16PrecisionForFp32(true);
    m_interpreter->SetNumThreads(numThreads);
}

void TFLiteMetricNet::predict(float feat1[], float feat2[], PredictResultMetricNet *res) {
    // get input & output layer of tflite model
    float *inputLayer1 = m_interpreter->typed_input_tensor<float>(0);
//    float *inputLayer2 = m_interpreter->typed_input_tensor<float>(1);
    float *outputLayer = m_interpreter->typed_output_tensor<float>(0);

    // merge input
    float feat[BATCH_SIZE * EMBEDDING_SIZE + EMBEDDING_SIZE];
    for (int i = 0; i < EMBEDDING_SIZE; i++){
        feat[i] = feat1[i];
    }
    for (int i = 0; i < BATCH_SIZE * EMBEDDING_SIZE; i++) {
        feat[i + EMBEDDING_SIZE] = feat2[i];
    }

    // copy the input image to input layer
    memcpy(inputLayer1, feat, (BATCH_SIZE * EMBEDDING_SIZE + EMBEDDING_SIZE) * sizeof(float));
//    memcpy(inputLayer1, feat1, EMBEDDING_SIZE * sizeof(float));
//    memcpy(inputLayer2, feat2, BATCH_SIZE * EMBEDDING_SIZE * sizeof(float));

    // compute model instance
    if (m_interpreter->Invoke() != kTfLiteOk) {
        printf("Error invoking detection model");
    } else{
        for (int i = 0; i < BATCH_SIZE; i++){
            res[i].results[0] = outputLayer[i];
        }

    }

}