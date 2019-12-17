#include "file_reader.h"

namespace onml {
std::vector<float>
read_y(std::string filename)
{
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    throw std::runtime_error("An error occurred while opening file " +
                             filename + ".");
  }
  std::string line;
  std::vector<float> y;
  while (getline(ifs, line)) {
    float f = std::stof(line);
    y.emplace_back(f);
  }
  return y;
}

std::vector<Eigen::VectorXf>
read_X(std::string filename)
{
  std::ifstream ifs(filename);
  if (ifs.fail()) {
    throw std::runtime_error("An error occurred while opening file " +
                             filename + ".");
  }
  std::string line;
  std::vector<Eigen::VectorXf> X;
  while (getline(ifs, line)) {
    std::istringstream iss(line);
    std::string field;
    std::vector<float> x;
    while (getline(iss, field, ' ')) {
      float f = std::stof(line);
      x.emplace_back(f);
    }
    X.emplace_back(Eigen::Map<Eigen::VectorXf>(x.data(), x.size()));
  }
  return X;
}
}
