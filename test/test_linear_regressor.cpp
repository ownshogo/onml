#include <catch.hpp>
#include <Eigen/Core>
#include <linear_regressor.h>

TEST_CASE("learn") {
	auto l = linear_regressor(1, 0.01);
	auto x1 = Eigen::VectorXf::Zero(1);
	auto y1 = 10.;
	auto x2 = Eigen::VectorXf::Ones(1);
	auto y2 = 100.;
	for (auto i = 0u; i < 1000000; ++i) {
		l.fit(x1, y1);
		l.fit(x2, y2);
	}
	REQUIRE(l.predict(x1) == Approx(y1).margin(0.01));
	REQUIRE(l.predict(x2) == Approx(y2).margin(0.01));
}
