#include "linear_classifier.h"

float
sigmoid(float z)
{
  return 1 / (1 + std::exp(-z));
}

namespace onml {
linear_classifier::linear_classifier(std::size_t dim,
                                     std::unique_ptr<loss> loss_func,
                                     std::unique_ptr<optimizer> opt)
{
  this->bias = 0.;
  this->weight = Eigen::VectorXf::Zero(dim);
  this->loss_func = std::move(loss_func);
  this->opt = std::move(opt);
}

void
linear_classifier::fit(const Eigen::VectorXf& x, bool y)
{
  float yf = y ? 1. : 0.;
  float yhat = this->predict_proba(x);
  float bias_gradient = this->loss_func->gradient(yf, yhat);
  this->bias = this->opt->next_bias(this->bias, bias_gradient);
  this->weight = this->opt->next_weights(this->weight, bias_gradient * x);
}

float
linear_classifier::predict_proba(const Eigen::VectorXf& x) const
{
  float z = this->bias + x.dot(this->weight);
  return sigmoid(z);
}
}
