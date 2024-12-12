# ONNXRuntime CXX
Cross-platform neural network inference based on OnnxRuntime C++
## License
[AGPL v3.0](LICENSE).<br>
Copyright &copy; 2024 [Hieu Pham](https://github.com/hieupth). All rights reserved.


## Structure
```
- libs
    - <third_party>
        - include
        - lib
- include
    - ortcxx
        - model.h
        - pipeline.h
- src
    - headers
        - model.h
        - pipeline.h
    - modules
        - model
            - model.cpp
        - pipeline
            - pipeline.cpp
```

`Model.cpp` includes the implementation of the `Model` class for `loading` and `running` using the ONNX Runtime C++ API.

`pipeline.cpp` includes the `Pipeline` class provides methods for preprocessing, postprocessing, and performing inference. The `inference` method raises an error if not overridden.

## Initializing the `Model` Class

To initialize the `Model` class, you need to provide the model path, configuration settings, available providers, and a boolean flag indicating a encrypted model. 

API
```
Model(
    <path_to_model>,
    <config>,
    <providers>,
    <is_Encrypted>
)
```

Here is an example based on the `hello.cpp` file:
```cpp
#include <ortcxx/model.h>

using namespace ortcxx::model;

// some custom config
std::map<std::string, std::any> config;
config["parallel"] = false;
config["inter_ops_threads"] = 1;
config["intra_ops_threads"] = 1;
config["graph_optimization_level"] = 1;
config["<model_name>.onnx"] = "<model_name>";

auto providers = Ort::GetAvailableProviders();
Model model = Model(modelPath, config, providers, false);
```


This code snippet demonstrates how to set up the necessary configuration and initialize the `Model` class with the specified parameters.


## Initialize the `Pipeline` Class
To initialize the `Pipeline` class, you need to provide a pointer to an instance of the `Model` class.

API
```
Pipeline(
    Model* model
)
```

Here is an example based on the `pipeline.cpp` file:
```cpp
#include <ortcxx/pipeline.h>
#include <ortcxx/model.h>

using namespace ortcxx::model;
using namespace ortcxx::pipeline;

// Assuming model is already initialized
Model* model = new Model(modelPath, config, providers, false);

Pipeline pipeline = Pipeline(model);

// Example input
Ort::Value* input = ...;

// Preprocess the input
Ort::Value* preprocessedInput = pipeline.preprocess(input);

// Perform inference
try {
    Ort::Value* output = pipeline.inference(preprocessedInput);
} catch (const std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
}

// Postprocess the output
Ort::Value* finalOutput = pipeline.postprocess(output);
```

This code snippet demonstrates how to set up the necessary configuration, initialize the `Pipeline` class with the specified parameters, and use its methods for preprocessing, inference, and postprocessing.

## WARNING
Feature UPDATE:
1. Initial Model not need to include `providers` because not used
2. Current error `Segmentation Fault` when runing `./pipeline`