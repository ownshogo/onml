#include "optimizer.h"

sgd::sgd(float learning_rate) : learning_rate(learning_rate) {}

float sgd::next_bias(float current_bias, float y_gradient) {
	return current_bias - this->learning_rate * y_gradient;
}

Eigen::VectorXf sgd::next_weights(const Eigen::VectorXf &current_weights, const Eigen::VectorXf &weights_gradient) {
	return current_weights - this->learning_rate * weights_gradient;
}

adagrad::adagrad(float initial_learning_rate, const std::size_t dim) : initial_learning_rate(initial_learning_rate) {
	this->bias_learning_rate = 0.;
	this->weights_learning_rate = Eigen::ArrayXf::Zero(dim);
	this->bias_r = 1e-8;
	this->weights_r = Eigen::ArrayXf::Constant(dim, 1e-8);
}

float adagrad::next_bias(float current_bias, float y_gradient) {
	this->bias_r += y_gradient * y_gradient;
	this->bias_learning_rate = this->initial_learning_rate / std::sqrt(this->bias_r);
	return current_bias - this->bias_learning_rate * y_gradient;
}

Eigen::VectorXf adagrad::next_weights(const Eigen::VectorXf &current_weights, const Eigen::VectorXf &weights_gradient) {
	Eigen::ArrayXf wga = weights_gradient.array();
	this->weights_r += wga.square();
	this->weights_learning_rate = this->initial_learning_rate / this->weights_r.sqrt();
	return current_weights - (this->weights_learning_rate * wga).matrix();
}
