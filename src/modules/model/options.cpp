#include "model.h"
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace Ort;
using namespace ortcxx::model;


bool ModelOptions::appendVINO(
  optional<map<string, any>> options
) {
  OrtOpenVINOProviderOptions optionsVINO;
  if (options.has_value()) {
    ///Other options are: GPU_FP32, GPU_FP16, MYRIAD_FP16
    auto device = options.value().find("device_openvino"); 
    if (device != options.value().end()) {
      optionsVINO.device_type = any_cast<string>(device->second).c_str();
      std::cout << "OpenVINO device type is set to: " << optionsVINO.device_type << std::endl;
      this->_sessOptions.AppendExecutionProvider_OpenVINO(optionsVINO);
      return true;
    }
  }
  return false;
}

bool ModelOptions::appendNNAPI(
  optional<map<string, any>> options
) {
  // fine the flag in options
  bool flag = false;  
  if (options.has_value()) {
    auto _options = options.value();
    auto it = _options.find("nnapi_flags");
    
    // Key found
    if (it != _options.end()) {
      uint32_t nnapi_flags = std::any_cast<int>(it->second);
      try {
        // try to append
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(
        //   this->_sessOptions, 
        //   nnapi_flags
        // ));
        // set
        flag = true;
      } catch (exception& e) {
        cout << "Error: " << e.what() << endl;
      }
    }
  }
  return flag;
};

bool ModelOptions::appendCPU(
  optional<map<string, any>> options
) {
  return true;
};

bool ModelOptions::appendCoreML(
  optional<map<string, any>> options
) {
  // fine the flag in options
  bool flag = false;  
  if (options.has_value()) {
    auto _options = options.value();
    auto it = _options.find("coreml_flags");
    
    // Key found
    if (it != _options.end()) {
      uint32_t coreml_flags = std::any_cast<int>(it->second);
      try {
        // try to append
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
        //   this->_sessOptions, 
        //   coreml_flags
        // ));
        // set
        flag = true;
      } catch (exception& e) {
        cout << "Error: " << e.what() << endl;
      }
    }
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

bool ModelOptions::appendCUDA(
  optional<map<string, any>> options
) { 
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
  optional<map<string, any>> options,
  optional<vector<string>> providers
) {
  // check if options are set
  if (options.has_value()) {
    auto _options = options.value();
    auto _begin = _options.begin();
    auto _end = _options.end();
    if (_options.find("parallel") != _end)
      try {
        this->_sessOptions.SetExecutionMode(any_cast<bool>(_options.at("parallel")) ? ORT_PARALLEL : ORT_SEQUENTIAL);
      } catch (bad_any_cast& e) {
        cout << "Invalid parrallel. Use default value." << endl;
      }
    if (_options.find("inter_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("inter_ops_threads"));
        if (threads > 0)
          this->_sessOptions.SetInterOpNumThreads(threads);
      } catch (bad_any_cast& e) {
        cout << "Invalid inter_ops_thread. Use default value." << endl;
      }
    if (_options.find("intra_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("intra_ops_threads"));
        if (threads > 0)
          this->_sessOptions.SetIntraOpNumThreads(threads);
      } catch(bad_any_cast& e) {
        cout << "Invalid intra_ops_thread. Use default value." << endl;
      }
    if (_options.find("graph_optimization_level") != _end)
      try {
        int graph = any_cast<int>(_options.at("graph_optimization_level"));
        switch (graph) {
          case 0: this->_sessOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL); break;
          case 1: this->_sessOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); break;
          case 2: this->_sessOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); break;
          case 3: this->_sessOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); break;
          default: break;
        }
      } catch (bad_any_cast& e) {
        cout << "Invalud graph_optimization_level. Use default value." << endl;
      }
    if (any_cast<bool>(_options.at("session.use_env_allocators")) == true)
      try {
        this->_sessOptions.AddConfigEntry("kOrtSessionOptionsConfigUseEnvAllocators", "1");
      } catch (bad_any_cast& e) {
        cout << "Invalid session_options. Use default value." << endl;
      }
  }

  // check if providers are set
  vector<string> AVAILABLE_PROVIDERS;
  if (providers.has_value()) {
    AVAILABLE_PROVIDERS = providers.value();
  }
  else {
    AVAILABLE_PROVIDERS = Ort::GetAvailableProviders();
  }

  // get the first provider
  auto providerName =  AVAILABLE_PROVIDERS.front();
  
  // Ort::SessionOptions* pSessionOptions = &sessionOptions;
  if (providerName == "CUDAExecutionProvider") {
    this->appendCUDA(options);
  } 
  else if (providerName == "OpenVINOExecutionProvider") {
    this->appendVINO(options);
  }
  else if (providerName == "NnapiExecutionProvider") {
    this->appendNNAPI(options);
  }
  else if (providerName == "CoreMLExecutionProvider") {
    this->appendCoreML(options);
  }
  else {
    this->appendCPU(options);
  }
  return this->_sessOptions.Clone();
}