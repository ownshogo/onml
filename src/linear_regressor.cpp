#include "linear_regressor.h"

linear_regressor::linear_regressor(std::size_t dim, std::unique_ptr<loss> loss_func, std::unique_ptr<optimizer> opt) {
	this->bias = 0.;
	this->weight = Eigen::VectorXf::Zero(dim);
	this->loss_func = std::move(loss_func);
	this->opt = std::move(opt);
}

float linear_regressor::predict(const Eigen::VectorXf &x) const {
	return bias + x.dot(this->weight);
}

void linear_regressor::fit(const Eigen::VectorXf &x, const float y) {
	float yhat = this->predict(x);
	float y_gradient = this->loss_func->gradient(y, yhat);
	this->bias = this->opt->next_bias(this->bias, y_gradient);
	this->weight = this->opt->next_weights(this->weight, y_gradient * x);
}
