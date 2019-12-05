#ifndef SRC_LINEAR_REGRESSOR_H_
#define SRC_LINEAR_REGRESSOR_H_

#include <vector>
#include <memory>
#include <Eigen/Core>
#include "loss.h"
#include "optimizer.h"

class linear_regressor {
public:
	linear_regressor(const std::size_t dim, std::unique_ptr<loss> loss_func, const std::unique_ptr<optimizer> opt);
	void fit(const Eigen::VectorXf &x, const float y);
	float predict(const Eigen::VectorXf &x) const;
private:
	float bias;
	Eigen::VectorXf weight;
	std::unique_ptr<loss> loss_func;
	std::unique_ptr<optimizer> opt;
};

#endif /* SRC_LINEAR_REGRESSOR_H_ */
