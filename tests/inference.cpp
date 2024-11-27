#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <ortcxx/model.h>
#include <ortcxx/pipeline.h>

using namespace std;
using namespace ortcxx::model;
using namespace ortcxx::pipeline;

int main(){
    std::cout << "Hello, from Cinnamon Runtime!\nAvailable Providers:" << std::endl;
    auto providers = Ort::GetAvailableProviders();
    for (std::string p : providers) {
        std::cout << "- " << p << std::endl;
    };
    map<string, any> c;
    c["parallel"] = false;
    c["inter_ops_threads"] = 1;
    c["intra_ops_threads"] = 1;
    c["graph_optimization_level"] = 1;

    // modelNm = "model_convert/extractor.onnx";
    std::string modelPath = "...";
    std::string modelName = "extractor.onnx";
    c["extractor.onnx"] = modelName;

    Model a = Model(modelPath, c, providers, false);

    vector<int64_t> inputShape = {1, 3, 256, 256};
    vector<float> input_ = vector<float>(1 * 3 * 256 * 256, 1.0);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memory_info, input_.data(), input_.size(), inputShape.data(), inputShape.size()
    );

    //turn input tensor to vector of Ort::Value
    vector<Ort::Value> inputs;
    inputs.push_back(std::move(inputTensor));

    vector<Ort::Value> outputs;

    Model* pointer_model = &a; 
    Pipeline p = Pipeline(pointer_model); // Pipeline object
    //init pointer
    Ort::Value* input_p = std::move(&inputTensor);
    // Ort::Value* output_p = &inputTensor;
    cout << "input_p: " << &inputTensor << endl;
    cout << "input_p: " << *input_p << endl;


    Ort::Value* output_p = p.preprocess(input_p);
    // Ort::Value output = p.inference(inputTensor);
    // Ort::Value result = p.postprocess(output);

    cout << "output: " << *output_p<< endl;
    cout << "output: " << output_p << endl;


    std::cout << "Head " << ": ";
    //! SEGMENT FAULT HERE
    auto info = input_p->GetTensorTypeAndShapeInfo(); 
    std::vector<int64_t> tensorShape = info.GetShape();

    for (int64_t dim : tensorShape) {
        std::cout << dim << " ";
    }   
    return 0;
}
