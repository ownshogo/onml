#ifndef SRC_LINEAR_REGRESSOR_H_
#define SRC_LINEAR_REGRESSOR_H_

#include <vector>

class linear_regressor {
public:
	linear_regressor(const std::size_t dim, const float learning_rate);
	void fit(const std::vector<float> &x, const float y);
	float predict(const std::vector<float> &x) const;
private:
	float bias;
	std::vector<float> weight;
	float learning_rate;
};

#endif /* SRC_LINEAR_REGRESSOR_H_ */
