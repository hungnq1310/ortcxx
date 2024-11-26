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
const auto AVAILABLE_PROVIDERS = GetAvailableProviders();

class ModelOptions
{
protected:
  SessionOptions _sessOptions;
  bool appendCPU(optional<map<string, any>> options,
                 Ort::SessionOptions* so);
  bool appendCUDA(optional<map<string, any>> options,
                  Ort::SessionOptions* so);
  bool appendVINO(optional<map<string, any>> options,
                  Ort::SessionOptions* so); 
  bool appendNNAPI(optional<map<string, any>> options,
                  Ort::SessionOptions* so);
  bool appendCoreML(optional<map<string, any>> options,
                  Ort::SessionOptions* so);
  
public:
  ModelOptions();
  SessionOptions getSessionOptions(optional<map<string, any>> options);
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
        const std::string& model, 
        std::shared_ptr<Ort::Env> env, 
        std::shared_ptr<Ort::Allocator> allocator, 
        const std::optional<std::map<std::string, std::any>> options,
        const optional<vector<string>> providers,
        bool isEncrypted
      ) {
        return std::shared_ptr<Model>(new Model(model, env, allocator, options, providers, isEncrypted));
      }

      Model(
          std::string model,
          const std::optional<std::map<std::string, std::any>> options,
          const optional<vector<string>> providers,
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
  //? why make this protected?
      Model(
          std::string model,
          std::shared_ptr<Ort::Env> env,
          std::shared_ptr<Ort::Allocator> allocator,
          const std::optional<std::map<std::string, std::any>> options,
          const optional<vector<string>> providers,
          bool isEncrypted
      );
  };
}; // namespace ortcxx::model
#endif