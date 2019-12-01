#include "linear_regressor.h"

linear_regressor::linear_regressor(const std::size_t dim, const float learning_rate) {
	this->bias = 0.;
	this->weight = std::vector<float>(dim);
	this->learning_rate = learning_rate;
}

float linear_regressor::predict(const std::vector<float> &x) const {
	float p = bias;
	for (auto i = 0u; i < x.size(); ++i) {
		p += x[i] * this->weight[i];
	}
	return p;
}

void linear_regressor::fit(const std::vector<float> &x, const float y) {
	float yhat = this->predict(x);
	float y_gradient = 2 * (yhat - y);
	this->bias -= this->learning_rate * y_gradient;
	for (auto i = 0u; i < x.size(); ++i) {
		this->weight[i] -= this->learning_rate * y_gradient * x[i];
	}
}
