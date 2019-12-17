#include <catch.hpp>
#include <Eigen/Core>
#include <random>
#include <chrono>
#include <algorithm>
#include "linear_regressor.h"
#include "file_reader.h"

using namespace onml;

TEST_CASE("learn") {
	auto l = linear_regressor(1, std::make_unique<squared_loss>(), std::make_unique<sgd>(0.01));
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

TEST_CASE("huber loss") {
	auto l = linear_regressor(1, std::make_unique<huber_loss>(10), std::make_unique<sgd>(0.01));
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

TEST_CASE("adagrad") {
	auto l = linear_regressor(1, std::make_unique<huber_loss>(10), std::make_unique<adagrad>(1, 1));
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

TEST_CASE("boston house price") {
	std::vector<Eigen::VectorXf> X = read_X("boston_X.data");
	std::vector<float> y = read_y("boston_y.data");
	std::size_t dim = X[0].size();
	auto lr = linear_regressor(dim, std::make_unique<squared_loss>(), std::make_unique<sgd>(1e-4));
	for (auto epoch = 0u; epoch < 1024; ++epoch) {
		std::random_device seedgen;
		std::mt19937 eng1(seedgen());
	   auto eng2 = eng1;
	   std::shuffle(X.begin(), X.end(), eng1);
	   std::shuffle(y.begin(), y.end(), eng2);
		for (auto i = 0u; i < X.size(); ++i) {
			lr.fit(X[i], y[i]);
		}
	}
	REQUIRE(lr.score(X, y) < 9*9);
}
