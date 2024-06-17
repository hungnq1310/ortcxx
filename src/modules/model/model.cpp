#include "model.h"

using namespace std;
using namespace Ort;

static SessionOptions getSessionOptions(
  const optional<map<string, any>> options = nullopt,
  const optional<map<string, optional<map<string, string>>>> providers = nullopt
) {
  Ort::SessionOptions sessionOptions = Ort::SessionOptions();
  if (options.has_value()) {
    auto _options = options.value();
    auto _begin = _options.begin();
    auto _end = _options.end();
    if (_options.find("parallel") != _end)
      try {
        sessionOptions.SetExecutionMode(any_cast<bool>(_options.at("parallel")) ? ORT_PARALLEL : ORT_SEQUENTIAL);
      } catch (bad_any_cast& e) {
        cout << "Invalid parrallel. Use default value." << endl;
      }
    if (_options.find("inter_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("inter_ops_threads"));
        if (threads > 0)
          sessionOptions.SetInterOpNumThreads(threads);
      } catch (bad_any_cast& e) {
        cout << "Invalid inter_ops_thread. Use default value." << endl;
      }
    if (_options.find("intra_ops_threads") != _end)
      try {
        int threads = any_cast<int>(_options.at("intra_ops_threads"));
        if (threads > 0)
          sessionOptions.SetIntraOpNumThreads(threads);
      } catch(bad_any_cast& e) {
        cout << "Invalid intra_ops_thread. Use default value." << endl;
      }
    if (_options.find("graph_optimization_level") != _end)
      try {
        int graph = any_cast<int>(_options.at("graph_optimization_level"));
        switch (graph) {
          case 0: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL); break;
          case 1: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC); break;
          case 2: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED); break;
          case 3: sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); break;
          default: break;
        }
      } catch (bad_any_cast& e) {
        cout << "Invalud graph_optimization_level. Use default value." << endl;
      }
  }
  if (providers.has_value()) {
    auto _providers = providers.value();
  }
  return sessionOptions;
}