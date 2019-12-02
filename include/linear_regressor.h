#ifndef SRC_LINEAR_REGRESSOR_H_
#define SRC_LINEAR_REGRESSOR_H_

#include <vector>
#include <Eigen/Core>

class linear_regressor {
public:
	linear_regressor(const std::size_t dim, const float learning_rate);
	void fit(const Eigen::VectorXf &x, const float y);
	float predict(const Eigen::VectorXf &x) const;
private:
	float bias;
	Eigen::VectorXf weight;
	float learning_rate;
};

#endif /* SRC_LINEAR_REGRESSOR_H_ */
