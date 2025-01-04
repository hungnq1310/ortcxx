#ifndef __ORTCXX_PIPELINE_H__
#define __ORTCXX_PIPELINE_H__

#include <string>
#include <ortcxx/model.h>
#include <onnxruntime_cxx_api.h>
#include <ortcxx/pipeline.h>

using namespace ortcxx::model;
using namespace std;
namespace ortcxx::pipeline {

class Pipeline {

    private:
        std::chrono::steady_clock::time_point sessionClock;
        std::thread gc;
        std::mutex clockMutex;
        bool stopGCFlag = false;
        void garbageCollector();
        int sessionDuration = 500;

    public:
        Pipeline() = default;
        Pipeline(shared_ptr<Model> model);
        ~Pipeline();

        virtual void preprocess() {}; 
        virtual void postprocess() {}; 
        virtual void inference() {};        

    protected:
        shared_ptr<Model> model;
        void updateSessionClock();
        float getSessionClock();
        void startGC();
        void stopGC();
    };

} // namespace pipeline

#endif // __ORTCXX_PIPELINE_H__