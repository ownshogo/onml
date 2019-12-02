#include "linear_regressor.h"

linear_regressor::linear_regressor(const std::size_t dim, const float learning_rate, std::unique_ptr<loss> loss_func) : learning_rate(learning_rate) {
	this->bias = 0.;
	this->weight = Eigen::VectorXf::Zero(dim);
	this->loss_func = std::move(loss_func);
}

float linear_regressor::predict(const Eigen::VectorXf &x) const {
	return bias + x.dot(this->weight);
}

void linear_regressor::fit(const Eigen::VectorXf &x, const float y) {
	float yhat = this->predict(x);
	float y_gradient = this->loss_func->gradient(y, yhat);
	this->bias -= this->learning_rate * y_gradient;
	this->weight -= this->learning_rate * y_gradient * x;
}
