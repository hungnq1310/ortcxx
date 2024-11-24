#include "model.h"
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace Ort;
using namespace ortcxx::model;

ModelOptions::ModelOptions(optional<map<string, any>> options)
{
  this->_sessOptions = SessionOptions();
};


bool ModelOptions::appendVINO(optional<map<string, any>> options)
{
  std::unordered_map<std::string, std::string> openVINOOptions;
  for (auto& pair : options.value()) {
    openVINOOptions[pair.first] = pair.second;
  }
  this->_sessOptions.AppendExecutionProvider("OpenVINO", openVINOOptions);
  return true;
};

bool ModelOptions::appendNNAPI(optional<map<string, any>> options)
{
  for (auto& pair : options.value()) {
    if (pair.first == "nnapi_flags" && pair.second == 0)
    {
      this->_sessOptions.AppendExecutionProvider_Nnapi(pair.second);
      return true;
    }
  }
  return false;
};

bool ModelOptions::appendCPU(optional<map<string, any>> options)
{
  return true;
};

bool ModelOptions::appendCoreML(optional<map<string, any>> options)
{
  bool flag = false;
  for (auto& pair : options.value()) {
    if (pair.first == "coreml_flags" && pair.second == 0)
    {
      this->_sessOptions.AppendExecutionProvider_CoreML(pair.second);
      flag = true;
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

bool ModelOptions::appendCUDA(optional<map<string, any>> options)
{ 
  // init
  OrtCUDAProviderOptionsV2* cudaOptions = nullptr;
  // create CUDA provider options
  checkStatusCUDA(Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions));
  vector<const char*> keys;
  vector<const char*> values;
  for (auto& pair : options.value()) {
      keys.push_back(pair.first.c_str());
      values.push_back(pair.second.c_str());
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
  }

  auto providerName =  AVAILABLE_PROVIDERS.front();
  auto _begin = AVAILABLE_PROVIDERS.begin();
  auto _end = AVAILABLE_PROVIDERS.end();

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
  
  return this->_sessOptions;
}