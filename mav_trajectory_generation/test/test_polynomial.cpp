/*
 * Copyright (c) 2016, Markus Achtelik, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Michael Burri, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Helen Oleynikova, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Rik Bähnemann, ASL, ETH Zurich, Switzerland
 * Copyright (c) 2016, Marija Popovic, ASL, ETH Zurich, Switzerland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <iostream>
#include <random>

#include <eigen-checks/gtest.h>

#include <mav_trajectory_generation/motion_defines.h>
#include <mav_trajectory_generation/polynomial.h>

using namespace mav_trajectory_generation;

const double kSamplingInterval = 1.0e-3;
const double kEqualityResolution = 1.0e-2;
const int kDerivative = derivative_order::POSITION;

void findMinMaxBySampling(const Polynomial& polynomial, double t_1, double t_2,
                          double* t_min, double* t_max, double* min,
                          double* max) {
  double t = t_1;
  *min = std::numeric_limits<double>::max();
  *max = std::numeric_limits<double>::lowest();
  while (t <= t_2) {
    double candidate = polynomial.evaluate(t, kDerivative);
    if (candidate < *min) {
      *min = candidate;
      *t_min = t;
    }
    if (candidate > *max) {
      *max = candidate;
      *t_max = t;
    }
    t += kSamplingInterval;
  }
}

double createRandomDouble(double min, double max) {
  // No seed for repeatability.
  return (max - min) * (static_cast<double>(std::rand()) /
                        static_cast<double>(RAND_MAX)) +
         min;
}

bool approxEqual(double x_1, double x_2) {
  double dist = std::abs(x_1 - x_2);
  return dist < kEqualityResolution;
}

TEST(PolynomialTest, FindMinMax) {
  const double kTMin = -100.0;
  const double kTMax = 100.0;
  const double kCoeffMin = -100.0;
  const double kCoeffMax = 100.0;

  std::srand(1234567);
  static int num_failures = 0;
  const int kNumPolynomials = 1e3;
  for (size_t i = 0; i < kNumPolynomials; i++) {
    // Create random polynomial.
    int num_coeffs = std::rand() % (Polynomial::kMaxN - 1) + 1;
    Eigen::VectorXd coeffs(num_coeffs);
    for (size_t i = 0; i < num_coeffs; i++) {
      coeffs[i] = createRandomDouble(kCoeffMin, kCoeffMax);
    }
    Polynomial p(num_coeffs, coeffs);

    // Calculate minimum and maximum.
    double t_min_sampling, t_max_sampling, min_sampling, max_sampling,
        t_min_computing, t_max_computing, min_computing, max_computing;
    double t_1 = createRandomDouble(kTMin, kTMax);
    double t_2 = createRandomDouble(t_1, kTMax);
    findMinMaxBySampling(p, t_1, t_2, &t_min_sampling, &t_max_sampling,
                         &min_sampling, &max_sampling);
    if (!p.findMinMax(t_1, t_2, kDerivative, &t_min_computing, &t_max_computing,
                      &min_computing, &max_computing)) {
      std::cout << "Failed to compute roots for " << coeffs.transpose()
                << std::endl;
      num_failures++;
      continue;
    }
    if (!approxEqual(t_max_sampling, t_max_computing) ||
        !approxEqual(t_min_sampling, t_min_computing)) {
      std::cout << "t_min_sampling: " << t_min_sampling << std::endl;
      std::cout << "t_min_computing: " << t_min_computing << std::endl;
      std::cout << "min_sampling: " << min_sampling << std::endl;
      std::cout << "min_computing: " << min_computing << std::endl;

      std::cout << "t_max_sampling: " << t_max_sampling << std::endl;
      std::cout << "t_max_computing: " << t_max_computing << std::endl;
      std::cout << "max_sampling: " << max_sampling << std::endl;
      std::cout << "max_computing: " << max_computing << std::endl;
    }
    EXPECT_TRUE(approxEqual(t_min_sampling, t_min_computing));
    EXPECT_TRUE(approxEqual(t_max_sampling, t_max_computing));
  }
  std::cout << "Failed to compute minimum for " << num_failures << " / "
            << kNumPolynomials << " polynomials." << std::endl;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
