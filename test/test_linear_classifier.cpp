#include <catch.hpp>
#include <Eigen/Core>
#include "linear_classifier.h"

using namespace onml;

TEST_CASE("learn") {
	auto l = linear_classifier(1, std::make_unique<logistic_loss>(), std::make_unique<sgd>(0.01));
	auto x1 = Eigen::VectorXf::Zero(1);
	auto y1 = true;
	auto x2 = Eigen::VectorXf::Ones(1);
	auto y2 = false;
	for (auto i = 0u; i < 1000000; ++i) {
		l.fit(x1, y1);
		l.fit(x2, y2);
	}
	REQUIRE(l.predict_proba(x1) == Approx(y1).margin(0.01));
	REQUIRE(l.predict_proba(x2) == Approx(y2).margin(0.01));
}

TEST_CASE("adagrad") {
	auto l = linear_classifier(1, std::make_unique<logistic_loss>(), std::make_unique<adagrad>(1, 1));
		auto x1 = Eigen::VectorXf::Zero(1);
		auto y1 = true;
		auto x2 = Eigen::VectorXf::Ones(1);
		auto y2 = false;
		for (auto i = 0u; i < 1000000; ++i) {
			l.fit(x1, y1);
			l.fit(x2, y2);
		}
		REQUIRE(l.predict_proba(x1) == Approx(y1).margin(0.01));
		REQUIRE(l.predict_proba(x2) == Approx(y2).margin(0.01));
}
