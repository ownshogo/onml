#include "linear_regressor.h"

namespace onml {
linear_regressor::linear_regressor(std::size_t dim,
                                   std::unique_ptr<loss> loss_func,
                                   std::unique_ptr<optimizer> opt)
{
  this->bias = 0.;
  this->weight = Eigen::VectorXf::Zero(dim);
  this->loss_func = std::move(loss_func);
  this->opt = std::move(opt);
}

float
linear_regressor::predict(const Eigen::VectorXf& x) const
{
  return bias + x.dot(this->weight);
}

void
linear_regressor::fit(const Eigen::VectorXf& x, const float y)
{
  float yhat = this->predict(x);
  float y_gradient = this->loss_func->gradient(y, yhat);
  this->bias = this->opt->next_bias(this->bias, y_gradient);
  this->weight = this->opt->next_weights(this->weight, y_gradient * x);
}

float
linear_regressor::score(const std::vector<Eigen::VectorXf>& X,
                        const std::vector<float>& y) const
{
  float score = 0.f;
  for (auto i = 0u; i < X.size(); ++i) {
    float yhat = this->predict(X[i]);
    score += this->loss_func->compute(yhat, y[i]);
  }
  return score / X.size();
}
}
