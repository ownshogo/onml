#ifndef INCLUDE_FILE_READER_H_
#define INCLUDE_FILE_READER_H_

#include <Eigen/Core>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace onml {
std::vector<float>
read_y(std::string filename);
std::vector<Eigen::VectorXf>
read_X(std::string filename);
}

#endif /* INCLUDE_FILE_READER_H_ */
