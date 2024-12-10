// #include "model.h"
#include <iostream>
#include <fstream>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <optional>
#include <map>
#include <any>
#include <ortcxx/model.h>

using namespace std;
using namespace Ort;
using namespace ortcxx::model;

#define encryptedKey "3!4%@Us287uEUo86^QSA%L"

ModelOptions::ModelOptions() {
    this->_sessOptions = SessionOptions();
}

Model::Model(
    const std::string modelPath,
    std::optional<std::map<std::string, std::any>> options,
    optional<vector<string>> providers,
    bool isEncrypted
) {
    // read the model path
    ifstream inputFile(modelPath, ios::binary);
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

    *this = Model(fileContent, fileSize, options, providers, isEncrypted);
}

Model::Model(
    const char * modelBuffer,
    size_t modelSize,
    std::optional<std::map<std::string, std::any>> options,
    optional<vector<string>> providers,
    bool isEncrypted
) {
    // Initialize the environment
    this->_env = make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");

    // Model options
    ModelOptions model_options;
    this->_sessionOptions = std::make_unique<Ort::SessionOptions>(model_options.getSessionOptions(options, providers));

    // Initialize the session
    this->_session = make_unique<Ort::Session>(*this->_env, modelBuffer, modelSize, *this->_sessionOptions);
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
    const std::string modelPath,
    std::shared_ptr<Ort::Env> env,
    std::shared_ptr<Ort::Allocator> allocator,
    std::optional<std::map<std::string, std::any>> options,
    optional<vector<string>> providers,
    bool isEncrypted
) {
    // Initialize the environment
    this->_env = env;
    this->_allocator = allocator;

    // Model options
    ifstream inputFile(modelPath, ios::binary);
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
    *this = Model(fileContent, fileSize, env, allocator, options, providers, isEncrypted);
}

Model::Model(
    const char *modelBuffer,
    size_t modelSize,
    std::shared_ptr<Ort::Env> env,
    std::shared_ptr<Ort::Allocator> allocator,
    std::optional<std::map<std::string, std::any>> options,
    optional<vector<string>> providers,
    bool isEncrypted
) {
    // Initialize the environment
    this->_env = env;
    //! This allocator is used for input and output names
    this->_allocator = allocator;

    // Model options
    ModelOptions model_options;
    this->_sessionOptions = std::make_unique<Ort::SessionOptions>(model_options.getSessionOptions(options, providers));

    if (this->_sessionOptions.find("kOrtSessionOptionsConfigUseEnvAllocators") == nullptr){
        throw runtime_error("Share `Env` was found but config `session.use_env_allocators` has not been set!!!");
    }

    // Initialize the session
    this->_session = make_unique<Ort::Session>(*this->_env, modelBuffer, modelSize, *this->_sessionOptions);
    this->_session
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
    auto inputNames = this->inputNames;
    auto outputNames = this->outputNames;
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

    if (this->_session == nullptr) {
        throw runtime_error("Session is not initialized");
    }

    try {   
        vector<Ort::Value> outputVector = this->_session->Run(runOptions, inputNames.data(), inputs.data(), inputNames.size(), outputNames.data(), outputNames.size());
        this->isRunned = true;
        return make_shared<vector<Ort::Value>>(move(outputVector));
        }
    catch (Ort::Exception& exception) {
        cout << "Error: " << exception.what() << endl;
        }
    return nullptr;
}