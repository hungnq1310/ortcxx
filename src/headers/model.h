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
#include <ortcxx/model.h>

using namespace std;
using namespace Ort;

namespace ortcxx::model
{

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
  ModelOptions();
  SessionOptions getSessionOptions(optional<map<string, any>> options, optional<vector<string>> providers);
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
    std::unique_ptr<Ort::SessionOptions> _sessionOptions;
    bool isRunned = false;

  public:
    bool isRunnedModel() {
      return this->isRunned;
    };

    static std::shared_ptr<Model> create(
      const std::string& modelPath, 
      std::shared_ptr<Ort::Env> env, 
      std::optional<std::map<std::string, std::any>> options,
      optional<vector<string>> providers,
      bool isEncrypted
    ) {
      return std::shared_ptr<Model>(new Model(modelPath, env, allocator, options, providers, isEncrypted));
    }

    static std::shared_ptr<Model> create(
      const char * modelBuffer,
      size_t modelSize,
      std::shared_ptr<Ort::Env> env, 
      std::optional<std::map<std::string, std::any>> options,
      optional<vector<string>> providers,
      bool isEncrypted
    ) {
      return std::shared_ptr<Model>(new Model(modelBuffer, modelSize, env, allocator, options, providers, isEncrypted));
    }

    Model(
      const std::string modelPath,
      std::optional<std::map<std::string, std::any>> options,
      optional<vector<string>> providers,
      bool isEncrypted
    );

    Model(
      const char * modelBuffer,
      size_t modelSize,
      std::optional<std::map<std::string, std::any>> options,
      optional<vector<string>> providers,
      bool isEncrypted
    );

    Model(
      const std::string modelPath,
      std::shared_ptr<Ort::Env> env,
      std::optional<std::map<std::string, std::any>> options,
      optional<vector<string>> providers,
      bool isEncrypted
    );

    Model(
      const char * modelBuffer,
      size_t modelSize,
      std::shared_ptr<Ort::Env> env,
      std::optional<std::map<std::string, std::any>> options,
      optional<vector<string>> providers,
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

  };
}; // namespace ortcxx::model
#endif