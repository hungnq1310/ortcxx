#include "model.h"
#include <onnxruntime_cxx_api.h>
#include <CL/cl2.hpp>

using namespace std;
using namespace Ort;
using namespace ortcxx::model;


bool ModelOptions::appendVINO(
  optional<map<string, any>> options,
  std::unique_ptr<Ort::SessionOptions> so
)
{
  OrtOpenVINOProviderOptions optionsVINO;
  optionsVINO.device_type = options.find("device_openvino"); //Another option is: GPU_FP16
  auto ocl_instance = std::make_shared<OpenCL>();
  optionsVINO.context = (void *) ocl_instance->_context.get() ; 
  std::cout << "OpenVINO device type is set to: " << options.device_type << std::endl;
  so->AppendExecutionProvider_OpenVINO(options);
};

bool ModelOptions::appendNNAPI(
  optional<map<string, any>> options,
  std::unique_ptr<Ort::SessionOptions> so
) {
  // fine the flag in options
  bool flag = false;  
  auto it = options.find("coreml_flags");
  
  // Key found
  uint32_t nnapi_flags = std::any_cast<int>(it->second);
  try{
    // try to append
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(&so, nnapi_flags));
    //set
    flag = true;
  } catch (exception& e) {
    cout << "Error: " << e.what() << endl;
  }
  return flag;
};

bool ModelOptions::appendCPU(
  optional<map<string, any>> options,
  std::unique_ptr<Ort::SessionOptions> so
) {
  return true;
};

bool ModelOptions::appendCoreML(optional<map<string, any>> options, std::unique_ptr<Ort::SessionOptions> so)
{
  // fine the flag in options
  bool flag = false;  
  auto it = options.find("coreml_flags");
  
  // Key found
  uint32_t coreml_flags = std::any_cast<int>(it->second);
  try{
    // try to append
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(&so, coreml_flags));
    //set
    flag = true;
  } catch (exception& e) {
    cout << "Error: " << e.what() << endl;
  }
  return flag;
};

void checkStatusCUDA(OrtStatus* status) {
  if (status != nullptr) {
    cout << "Error: " << Ort::GetApi().GetErrorMessage(status) << endl;
    Ort::GetApi().ReleaseStatus(status);
    throw std::runtime_error("Error occurred during CUDA provider options creation or update.");
  }
}

bool ModelOptions::appendCUDA(optional<map<string, any>> options, std::unique_ptr<Ort::SessionOptions> so)
{ 
  // init
  OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
  // create CUDA provider options
  checkStatusCUDA(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
  vector<const char*> keys;
  vector<const char*> values;
  for (auto& pair : options.value()) {
      keys.push_back(pair.first.c_str());
      values.push_back(any_cast<const char*>(pair.second));
  }
  // update CUDA provider options with options
  checkStatusCUDA(Ort::GetApi().UpdateCUDAProviderOptions(cudaOptions, keys.data(), values.data(), 1));
  // map CUDA provider options to session options
  checkStatusCUDA(Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(this->_sessOptions, cudaOptions));
  // release CUDA provider options
  Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);
  return true;
};


SessionOptions ModelOptions::getSessionOptions(
  const optional<map<string, any>> options
) {

  Ort::SessionOptions sessionOptions = Ort::SessionOptions();

  if (options.has_value()) {
    auto _options = options.value();
    auto _begin = _options.begin();
    auto _end = _options.end();
    if (_options.find("parallel") != _end)
      try {
        sessionOptions.SetExecutionMode(any_cast<bool>(_options.at("parallel")) ? ORT_PARALLEL : ORT_SEQUENTIAL);
      } catch (bad_any_cast& e) {
        cout << "Invalid parrallel. Use default value." << endl;
      }
    if (_options.find("inter_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("inter_ops_threads"));
        if (threads > 0)
          sessionOptions.SetInterOpNumThreads(threads);
      } catch (bad_any_cast& e) {
        cout << "Invalid inter_ops_thread. Use default value." << endl;
      }
    if (_options.find("intra_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("intra_ops_threads"));
        if (threads > 0)
          sessionOptions.SetIntraOpNumThreads(threads);
      } catch(bad_any_cast& e) {
        cout << "Invalid intra_ops_thread. Use default value." << endl;
      }
    if (_options.find("graph_optimization_level") != _end)
      try {
        int graph = any_cast<int>(_options.at("graph_optimization_level"));
        switch (graph) {
          case 0: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL); break;
          case 1: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); break;
          case 2: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); break;
          case 3: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); break;
          default: break;
        }
      } catch (bad_any_cast& e) {
        cout << "Invalud graph_optimization_level. Use default value." << endl;
      }
  }

  auto providerName =  AVAILABLE_PROVIDERS.front();
  auto _begin = AVAILABLE_PROVIDERS.begin();
  auto _end = AVAILABLE_PROVIDERS.end();

  auto pSessionOptions = std::make_unique<Ort::SessionOptions>(sessionOptions);
  if (providerName == "CUDAExecutionProvider") {
    this->appendCUDA(options, pSessionOptions);
  } 
  else if (providerName == "OpenVINOExecutionProvider") {
    this->appendVINO(options, pSessionOptions);
  }
  else if (providerName == "NnapiExecutionProvider") {
    this->appendNNAPI(options, pSessionOptions);
  }
  else if (providerName == "CoreMLExecutionProvider") {
    this->appendCoreML(options, pSessionOptions);
  }
  else {
    this->appendCPU(options, pSessionOptions);
  }
  return sessionOptions;
}