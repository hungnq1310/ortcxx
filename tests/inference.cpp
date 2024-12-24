#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <ortcxx/model.h>
#include <ortcxx/pipeline.h>
#include <fstream>

using namespace std;
using namespace ortcxx::model;
using namespace ortcxx::pipeline;

int main(){
    std::cout << "Hello, from Cinnamon Runtime!\nAvailable Providers:" << std::endl;
    auto providers = Ort::GetAvailableProviders();
    for (std::string p : providers) {
        std::cout << "- " << p << std::endl;
    };
    map<string, any> c;
    c["parallel"] = false;
    c["inter_ops_threads"] = 1;
    c["intra_ops_threads"] = 1;
    c["graph_optimization_level"] = 1;
    c["session.use_env_allocators"] = true;
    std::string modelPath = "...";

    // init env and allocator
    shared_ptr<Ort::Env> env = make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
    env->CreateAndRegisterAllocator(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault), {});

    //--------------------------------------------------------------------------------

    // Model a - Model file
    //test pipeline
    shared_ptr<Model> pointer_model1 = Model::create(
        modelPath, env, c, providers, false
    );
    //init pointer
    Pipeline pipeline1 = Pipeline(pointer_model1);
    printf("Address of pipeline object: %p\n", &pipeline1);

    //--------------------------------------------------------------------------------

    // Model b - Model buffer
    ifstream inputFile(modelPath, ios::binary);
    if (!inputFile.is_open()) {
        cerr << "Error reading file." << endl;
    }
    inputFile.seekg(0, inputFile.end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(0, inputFile.beg);
    char *fileContent = new char[fileSize];
    inputFile.read(fileContent, fileSize);
    inputFile.close();
    //test pipeline 1
    shared_ptr<Model> pointer_mode2 = Model::create(
        fileContent, fileSize, env, c, providers, false
    );
    Pipeline pipeline2 = Pipeline(pointer_mode2);
    printf("Address of pipeline object: %p\n", &pipeline2);

    //--------------------------------------------------------------------------------
    return 0;
}
