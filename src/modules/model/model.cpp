#include "model.h"

using namespace std;
using namespace Ort;
using namespace ortcxx::model;

#define encryptedKey "3!4%@Us287uEUo86^QSA%L"


Model::Model(
    std::string model,
    const std::optional<std::map<std::string, std::any>> options,
    const optional<map<string, optional<map<string, string>>>> providers,
    bool isEncrypted
) {
    // Initialize the environment
    this->_env = make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

    // Model options
    ModelOptions model_options(options);
    this->_sessionOptions = model_options.getSessionOptions(options);
  
    ifstream inputFile(model, ios::binary);
    if (!inputFile.is_open()) {
        cerr << "Error reading file." << endl;
    }

    // Read the file content
    inputFile.seekg(0, inputFile.end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, inputFile.beg);
    char *fileContent = new char[fileSize];
    inputFile.read(fileContent, fileSize);
    inputFile.close();

    this->_session = make_unique<Ort::Session>(*this->_env, fileContent, fileSize, *this->_sessionOptions);
    this->_allocator = make_shared<Ort::Allocator>(*this->_session, Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    for (size_t i = 0; i < this->_session->GetInputCount(); ++i) {
        Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(i, *this->_allocator);
        this->inputNames.push_back(inputName.release());
    }
    for (size_t i = 0; i < this->_session->GetOutputCount(); ++i) {
        Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(i, *this->_allocator);
        this->outputNames.push_back(outputName.release());
    }
}


Model::Model(
    std::string model,
    std::shared_ptr<Ort::Env> env,
    std::shared_ptr<Ort::Allocator> allocator,
    const std::optional<std::map<std::string, std::any>> options,
    const optional<map<string, any>> providers,
    bool isEncrypted
) {
    // Initialize the environment
    this->_env = env;
    this->_allocator = allocator;

    // Model options
    ModelOptions model_options(options);
    this->_sessionOptions = model_options.getSessionOptions(options);

  
    ifstream inputFile(model, ios::binary);
    if (!inputFile.is_open()) {
        cerr << "Error reading file." << endl;
    }

    // Read the file content
    inputFile.seekg(0, inputFile.end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, inputFile.beg);
    char *fileContent = new char[fileSize];
    inputFile.read(fileContent, fileSize);
    inputFile.close();

    // Initialize the session
    this->_session = make_unique<Ort::Session>(*this->_env, fileContent, fileSize, *this->_sessionOptions);
  
    for (size_t i = 0; i < this->_session->GetInputCount(); ++i) {
        Ort::AllocatedStringPtr inputName = this->_session->GetInputNameAllocated(i, *this->_allocator);
        this->inputNames.push_back(inputName.release());
    }

    for (size_t i = 0; i < this->_session->GetOutputCount(); ++i) {
        Ort::AllocatedStringPtr outputName = this->_session->GetOutputNameAllocated(i, *this->_allocator);
        this->outputNames.push_back(outputName.release());
    }
}


shared_ptr<vector<Ort::Value>> Model::run(
    const vector<Ort::Value>& inputs,
    shared_ptr<const char*> outputHead,
    const Ort::RunOptions& runOptions
) {
    if (inputs.size() != inputNames.size()) {
        throw runtime_error("Number of input values does not match the number of input names.");
    } 

    if (outputHead != nullptr) {
        bool found = false;
        for (const auto& name : outputNames) {
        if (std::strcmp(name, *outputHead) == 0) {
            found = true;
            break;
        }
        }
        if (found) {
        outputNames.clear();
        outputNames.push_back(*outputHead);
        }
    }

    if (this->_session == nullptr)
        throw runtime_error("Session is not initialized");

    if (this->_device == "CPU") {
        try {   
        vector<Ort::Value> outputVector = this->_session->Run(runOptions, inputNames.data(), inputs.data(), inputNames.size(), outputNames.data(), outputNames.size());
        this->isRunned = true;
        return make_shared<vector<Ort::Value>>(move(outputVector));
        }
        catch (Ort::Exception& exception) {
        cout << "Error: " << exception.what() << endl;
        }
    }
    else if (this->_device == "GPU") {
        //! FIX
        string deviceType = Model::mapProviderType[this->_device]; 
        Ort::MemoryInfo cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::MemoryInfo gpuMemoryInfo{deviceType.c_str(), OrtDeviceAllocator, 0, OrtMemTypeDefault};
        Ort::IoBinding ioBinding{*this->_session};
        
        for (size_t i = 0; i < inputNames.size(); ++i) {
            ioBinding.BindInput(inputNames[i], inputs[i]);
        }
        
        for (size_t i = 0; i < outputNames.size(); ++i) {
        ioBinding.BindOutput(outputNames[i], gpuMemoryInfo);
        }

        try {
        this->_session->Run(runOptions, ioBinding);
        vector<Ort::Value> outputTensor = ioBinding.GetOutputValues();
        this->isRunned = true;
        return make_shared<vector<Ort::Value>>(move(outputTensor));
        }
        catch (Ort::Exception& exception) {
        cout << "Error: " << exception.what() << endl;
        }
    }
    return nullptr;
}


future<shared_ptr<vector<Ort::Value>>> Model::runAsync(
    const vector<Ort::Value>& inputs, 
    shared_ptr<const char*> outputHead,
    const Ort::RunOptions runOptions){
    if (inputs.size() != inputNames.size()) {
        throw runtime_error("Number of input values does not match the number of input names.");
    }

    if (this->_session == nullptr)
        throw runtime_error("Session is not initialized");
    return async(launch::async, &Model::run, this, cref(inputs), outputHead, cref(runOptions));
}

//FIX
std::map<std::string, modelConfig> readConfig(const std::string& modelsDir) {
    std::map<std::string, modelConfig> modelConfigs;
    try {
        for (const auto& entry : std::filesystem::directory_iterator(modelsDir)) {
            if (entry.is_directory()) {
            std::string modelName = entry.path().filename().string();
            std::string yamlPath = (entry.path() / (modelName + ".yaml")).string();
                
            // Read model name
            if (!std::filesystem::exists(yamlPath)) {
                std::cerr << "Config file " << yamlPath << " does not exist." << std::endl;
                continue;
            }

            // Read yaml file
            YAML::Node config = YAML::LoadFile(yamlPath);

            // Options
            std::map<std::string, std::any> options;
            if (config["options"]) {
                options["parallel"] = config["options"]["parallel"].as<bool>();
                options["inter_ops_threads"] = config["options"]["inter_ops_threads"].as<int>();
                options["intra_ops_threads"] = config["options"]["intra_ops_threads"].as<int>();
                options["graph_optimization_level"] = config["options"]["graph_optimization_level"].as<int>();
            }

            // Providers
            std::map<std::string, std::optional<std::map<std::string, std::string>>> providers;
            if (config["providers"]) {
                for (const auto& provider : config["providers"]) {
                std::string providerName = provider.first.as<std::string>();
                if (provider.second.IsMap()) {
                std::map<std::string, std::string> providerOptions;
                for (const auto& option : provider.second) 
                providerOptions[option.first.as<std::string>()] = option.second.as<std::string>();
                providers[providerName] = providerOptions;
                } else
                providers[providerName] = std::nullopt;
                }
            }

            // File settings
            bool encryptedFile = false;
            if (config["file_settings"]) {
                encryptedFile = config["file_settings"]["encrypted_file"].as<bool>();
            }
            std::string modelFile;
            if (encryptedFile) 
                modelFile = (entry.path() / (modelName + ".enc")).string();
            else 
                modelFile = (entry.path() / (modelName + ".onnx")).string();

            // Save model config
            modelConfigs[modelName] = modelConfig{options, providers, encryptedFile, modelFile};
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading model configs: " << e.what() << std::endl;
    }
    return modelConfigs;
}