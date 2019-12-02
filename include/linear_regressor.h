#ifndef SRC_LINEAR_REGRESSOR_H_
#define SRC_LINEAR_REGRESSOR_H_

#include <vector>
#include <memory>
#include <Eigen/Core>
#include "loss.h"

class linear_regressor {
public:
	linear_regressor(const std::size_t dim, const float learning_rate, std::unique_ptr<loss> loss_func);
	void fit(const Eigen::VectorXf &x, const float y);
	float predict(const Eigen::VectorXf &x) const;
private:
	float bias;
	Eigen::VectorXf weight;
	const float learning_rate;
	std::unique_ptr<loss> loss_func;
};

#endif /* SRC_LINEAR_REGRESSOR_H_ */
