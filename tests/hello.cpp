#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <ortcxx/model.h>

using namespace std;
using namespace ortcxx::model;

int main(){
    std::cout << "Hello, from Cinnamon Runtime!\nAvailable Providers:" << std::endl;
    auto providers = Ort::GetAvailableProviders();
    for (std::string p : providers) {
        std::cout << "- " << p << std::endl;
    };
    map<string, any> c;
    ModelOptions b = ModelOptions(c);
    return 0;
}
