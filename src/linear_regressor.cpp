#include "linear_regressor.h"

linear_regressor::linear_regressor(const std::size_t dim, const float learning_rate) {
	this->bias = 0.;
	this->weight = Eigen::VectorXf::Zero(dim);
	this->learning_rate = learning_rate;
}

float linear_regressor::predict(const Eigen::VectorXf &x) const {
	return bias + x.dot(this->weight);
}

void linear_regressor::fit(const Eigen::VectorXf &x, const float y) {
	float yhat = this->predict(x);
	float y_gradient = 2 * (yhat - y);
	this->bias -= this->learning_rate * y_gradient;
	this->weight -= this->learning_rate * y_gradient * x;
}
