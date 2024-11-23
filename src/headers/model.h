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
#include <model.h>

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
  bool appendNNAPI(optional<map<string, any>> options);
  bool appendCoreML(optional<map<string, any>> options);
  
public:
  ModelOptions(optional<map<string, any>> options);
  ModelOptions(optional<map<string, any>> options, optional<map<string, any>> providers);
  SessionOptions getSessionOptions(optional<map<string, any>> options, optional<map<string, string>> providers);
};

class Model {
  protected:
      std::shared_ptr<Ort::Env> _env;
      std::shared_ptr<Ort::Allocator> _allocator;
      std::vector<const char*> inputNames;
      std::vector<const char*> outputNames;
      std::unique_ptr<Ort::Session> _session;
      std::unique_ptr<ModelOptions> _modelOptions;
      std::string _device;
      Ort::SessionOptions _sessionOptions;
      bool isRunned = false;

  public:
      bool isRunnedModel() {
          return this->isRunned;
      };

      Model(
          const std::string& model, 
          std::shared_ptr<Ort::Env> env, 
          std::shared_ptr<Ort::Allocator> allocator, 
          std::unique_ptr<ModelOptions> _modelOptions,
          const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers,
          bool isEncrypted
      );

      Model(
          std::string model,
          std::unique_ptr<ModelOptions> _modelOptions,
          const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers,
          bool isEncrypted
      );

      std::shared_ptr<std::vector<Ort::Value>> run(
          const std::vector<Ort::Value>& inputs,
          std::shared_ptr<const char*> outputHead = nullptr,
          const Ort::RunOptions& runOptions = Ort::RunOptions()
      );
      
      std::future<std::shared_ptr<std::vector<Ort::Value>>> runAsync(
          const std::vector<Ort::Value>& inputs,
          std::shared_ptr<const char*> outputHead = nullptr,
          const Ort::RunOptions runOptions = Ort::RunOptions()
      );

  protected:
      Model(
          std::string model,
          std::shared_ptr<Ort::Env> env,
          std::shared_ptr<Ort::Allocator> allocator,
          std::unique_ptr<ModelOptions> _modelOptions,
          const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers,
          bool isEncrypted
      );
      Model(
          std::string model,
          std::unique_ptr<ModelOptions> _modelOptions,
          const std::optional<std::map<std::string, std::optional<std::map<std::string, std::string>>>> providers,
          bool isEncrypted
      );
  };
};
#endif