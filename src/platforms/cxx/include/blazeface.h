// #include <opencv2/dnn.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

// using namespace cv;
// using namespace std;

// struct FaceResult {
// 	float score = 0.0;
// 	Rect bbox;
//     float keypoints[10] = {0.0};
// };

// class BlazeFace {
// public:
// 	BlazeFace();
//     ~BlazeFace();
//     int detect(Mat src, FaceResult *res);
//     // Methods
//     void init(const char *model, long modelSize);
// 	void init(const char *model);
// 	void init(const char *model, int numThreads);
//     static const int MAX_OUTPUT = 2304;
// 	float conf_threshold = 0.5;
// 	float nms_threshold = 0.3;
//     const int OUTPUT_WEIGHT = 16;
// private:
// 	// members
// 	const int INPUT_SIZE = 192;
// 	const int INPUT_CHANNELS = 3;
// 	float anchors[MAX_OUTPUT][2];
// 	char *m_modelBytes = nullptr;
// 	std::unique_ptr<tflite::FlatBufferModel> m_model;
// 	std::unique_ptr<tflite::Interpreter> m_interpreter;
// 	void preprocess(Mat input, Mat &input_data, float padding[]);
	
// };

#include "featurenet.h"
#include <iostream>
#include <cassert>
#include <cstring>

FeatureNet::FeatureNet(const std::string& model_path)
    : Pipeline(model_path) {
    // Additional initialization if needed
}

FeatureNet::~FeatureNet() {
    if (m_modelBytes != nullptr) {
        free(m_modelBytes);
        m_modelBytes = nullptr;
    }
}

void FeatureNet::initModel(const char* tfliteModel, long modelSize) {
    // Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
    m_modelBytes = (char*)malloc(sizeof(char) * modelSize);
    memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
    m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);
    assert(m_model != nullptr);

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*m_model, resolver);
    builder(&m_interpreter);
    assert(m_interpreter != nullptr);

    if (m_interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return;
    }
}

std::vector<float> FeatureNet::preprocess(const std::vector<float>& input) {
    // Default implementation: return the input as is
    std::cout << "FeatureNet Preprocessing..." << std::endl;
    return input;
}

std::vector<float> FeatureNet::postprocess(const std::vector<float>& input) {
    // Default implementation: return the input as is
    std::cout << "FeatureNet Postprocessing..." << std::endl;
    return input;
}