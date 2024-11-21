// #ifndef FEATURENET_H
// #define FEATURENET_H
// #include <math.h>
// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/calib3d.hpp>
// #include "tensorflow/lite/interpreter.h"
// #include "tensorflow/lite/kernels/register.h"
// #include "tensorflow/lite/model.h"
// #include "tensorflow/lite/optional_debug_tools.h"

// using namespace cv;
// using namespace std;

// const int EMBEDDING_SIZE=512;

// // The facial landmarks coordinate for face align
// const float SRC_MAP[5][5][2]={
//         // left
//         {{51.642, 50.115}, {57.617, 49.990}, {35.740, 69.007}, {51.157, 89.050}, {57.025, 89.702}},
//         // lef profile
//         {{45.031, 50.118}, {65.568, 50.872}, {39.677, 68.111}, {45.177, 86.190}, {64.246, 86.758}},
//         // frontal
//         {{39.730, 51.138}, {72.270, 51.138}, {56.000, 68.493}, {42.463, 87.010}, {69.537, 87.010}},
//         // right
//         {{46.845, 50.872}, {67.382, 50.118}, {72.737, 68.111}, {48.167, 86.758}, {67.236, 86.190}},
//         // right frontal
//         {{54.796, 49.990}, {60.771, 50.115}, {76.673, 69.007}, {55.388, 89.702}, {61.257, 89.050}}
// };
// void cropFace(Mat src, Mat& dst, float landmarks[][2], int out_height, int out_width);
// float calCosimilarity(float* com, float* ref);

// struct PredictResultFeatureNet {
// 	float embedding[EMBEDDING_SIZE];
// };

// class TFLiteFeatureNet {
// public:
// 	TFLiteFeatureNet();
//     ~TFLiteFeatureNet();
//     void predict(Mat src, PredictResultFeatureNet &res);
//     // Methods
//     void initModel(const char *model, long modelSize);
//     void initModel(const char *model);
//     void initModel(const char *model, int numThreads);
// 	static const int INPUT_SIZE = 112;
// private:
// 	// members
// 	const int INPUT_CHANNELS = 3;
// 	char *m_modelBytes = nullptr;
// 	unique_ptr<tflite::FlatBufferModel> m_model;
// 	unique_ptr<tflite::Interpreter> m_interpreter;
	
// };
// #endif

#ifndef FEATURENET_H
#define FEATURENET_H

#include "pipeline.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <memory>
#include <string>

class FeatureNet : public Pipeline {
public:
    FeatureNet(const std::string& model_path);
    ~FeatureNet();
    std::vector<float> preprocess(const std::vector<float>& input) override;
    std::vector<float> postprocess(const std::vector<float>& input) override;
    void initModel(const char* tfliteModel, long modelSize);

private:
    char* m_modelBytes = nullptr;
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;
};

#endif // FEATURENET_H