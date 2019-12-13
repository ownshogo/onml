#ifndef SRC_LINEAR_CLASSIFIER_H_
#define SRC_LINEAR_CLASSIFIER_H_

#include "loss.h"
#include "optimizer.h"
#include <Eigen/Core>
#include <cmath>
#include <memory>

namespace onml {
class linear_classifier
{
public:
  linear_classifier(std::size_t dim,
                    std::unique_ptr<loss> loss_func,
                    std::unique_ptr<optimizer> opt);
  void fit(const Eigen::VectorXf& x, bool y);
  float predict_proba(const Eigen::VectorXf& x) const;

private:
  float bias;
  Eigen::VectorXf weight;
  std::unique_ptr<loss> loss_func;
  std::unique_ptr<optimizer> opt;
};
}
#endif /* SRC_LINEAR_CLASSIFIER_H_ */
