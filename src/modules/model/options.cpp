#include "model.h"

using namespace std;
using namespace Ort;
using namespace ortcxx::model;

ModelOptions::ModelOptions(optional<map<string, any>> options)
{
  this->_sessOptions = SessionOptions();
};