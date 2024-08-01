#ifndef __ORTCXX_MODEL_H__
#define __ORTCXX_MODEL_H__

#include <any>
#include <map>
#include <thread>
#include <string>
#include <vector>
#include <future>
#include <optional>
#include <iostream>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace Ort;

namespace ortcxx::model
{
const auto AVAILABLE_PROVIDERS = GetAvailableProviders();

class ModelOptions
{
protected:
  SessionOptions _sessOptions;
  bool appendCPU(optional<map<string, any>> options);
  bool appendCUDA(optional<map<string, any>> options);
  bool appendVINO(optional<map<string, any>> options);

public:
  ModelOptions(optional<map<string, any>> options);
};

}
#endif