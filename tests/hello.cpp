#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <ortcxx/model.h>

using namespace std;
using namespace ortcxx::model;

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

    try {
        std::shared_ptr<std::vector<Ort::Value>> outputTensors = a.run(
            inputs,
            shared_ptr<const char*>(),
            Ort::RunOptions()
        );

        std::cout << "Output has : " << outputTensors->size() << " elements\n";
        for (size_t i = 0; i < outputTensors->size(); ++i) {
            std::cout << "Head " << i << ": ";
            auto info = outputTensors->at(i).GetTensorTypeAndShapeInfo();    
            std::vector<int64_t> tensorShape = info.GetShape();
            for (int64_t dim : tensorShape) {
                std::cout << dim << " ";
            }
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}
