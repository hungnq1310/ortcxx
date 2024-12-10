#include <iostream>
#include <stdexcept>
#include <ortcxx/pipeline.h>

using namespace ortcxx::model;
using namespace ortcxx::pipeline;

Pipeline::Pipeline(shared_ptr<Model> model){
    this->model = model;
    // updateSessionClock(); //* CURRENTLY NOT USED
}

Pipeline::~Pipeline() {
    // Release the session
    // this->session = nullptr;
    this->model = nullptr;
}


void Pipeline::updateSessionClock(){
    std::lock_guard<std::mutex> lock(clockMutex);
    this->sessionClock = std::chrono::steady_clock::now();
}

float Pipeline::getSessionClock(){
    std::lock_guard<std::mutex> lock(clockMutex);
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - this->sessionClock).count();
    return duration;
}


void Pipeline::garbageCollector(){
    while (!stopGCFlag)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Run every 5ms
        std::lock_guard<std::mutex> lock(clockMutex);
        auto currentTime = std::chrono::steady_clock::now();
       
        if (std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - this->sessionClock
        ).count() > this->sessionDuration){
            this->~Pipeline();
            
        }
    }
}

void Pipeline::startGC(){
    if (gc.joinable()){
        gc.join();
    }
    stopGCFlag = false;
    gc = std::thread(&Pipeline::garbageCollector, this);
}

void Pipeline::stopGC(){
    stopGCFlag = true;
}