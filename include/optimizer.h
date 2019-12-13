#ifndef SRC_OPTIMIZER_H_
#define SRC_OPTIMIZER_H_

#include <Eigen/Core>
#include <cmath>

class optimizer
{
public:
  virtual float next_bias(float current_bias, float bias_gradient) = 0;
  virtual Eigen::VectorXf next_weights(
    const Eigen::VectorXf& current_weights,
    const Eigen::VectorXf& weights_gradient) = 0;
  virtual ~optimizer() {}
};

class sgd : public optimizer
{
public:
  sgd(float learning_rate);
  float next_bias(float current_bias, float bias_gradient) override;
  Eigen::VectorXf next_weights(
    const Eigen::VectorXf& current_weights,
    const Eigen::VectorXf& weights_gradient) override;

private:
  const float learning_rate;
};

class adagrad : public optimizer
{
public:
  adagrad(float initial_learning_rate, const std::size_t dim);
  float next_bias(float current_bias, float bias_gradient) override;
  Eigen::VectorXf next_weights(
    const Eigen::VectorXf& current_weights,
    const Eigen::VectorXf& weights_gradient) override;

private:
  const float initial_learning_rate;
  float bias_learning_rate;
  Eigen::ArrayXf weights_learning_rate;
  float bias_r;
  Eigen::ArrayXf weights_r;
};
#endif /* SRC_OPTIMIZER_H_ */
