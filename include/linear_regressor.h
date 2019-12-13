#ifndef SRC_LINEAR_REGRESSOR_H_
#define SRC_LINEAR_REGRESSOR_H_

#include "loss.h"
#include "optimizer.h"
#include <Eigen/Core>
#include <memory>
#include <vector>

namespace onml {
class linear_regressor
{
public:
  linear_regressor(std::size_t dim,
                   std::unique_ptr<loss> loss_func,
                   std::unique_ptr<optimizer> opt);
  void fit(const Eigen::VectorXf& x, float y);
  float predict(const Eigen::VectorXf& x) const;

private:
  float bias;
  Eigen::VectorXf weight;
  std::unique_ptr<loss> loss_func;
  std::unique_ptr<optimizer> opt;
};
}
#endif /* SRC_LINEAR_REGRESSOR_H_ */
