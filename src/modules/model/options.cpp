#include "model.h"

using namespace std;
using namespace Ort;
using namespace ortcxx::model;

ModelOptions::ModelOptions(optional<map<string, any>> options)
{
  this->_sessOptions = SessionOptions();
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

