/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#pragma once

#include <gtsam/base/VectorSpace.h>

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace gtsam {

/**
 * QCQP representations default to the operations supplied by manifold traits.
 * Specializations allow Euclidean variables to participate without adding
 * optimization-specific operations to the general Eigen/vector-space traits.
 */
template <typename T, typename Enable = void>
struct QcqpTraits : traits<T> {};

/** Homogeneous D=1 representation [1; point] of a fixed-size column vector. */
template <int N, int Options, int MaxRows, int MaxCols>
struct QcqpTraits<Eigen::Matrix<double, N, 1, Options, MaxRows, MaxCols>,
                  std::enable_if_t<(N > 0)>> {
  using Point = Eigen::Matrix<double, N, 1, Options, MaxRows, MaxCols>;
  inline constexpr static int QcqpVectorDim = N + 1;

  /** Encode a Euclidean point as a homogeneous column vector. */
  template <int D = 1>
  static Matrix QcqpValue(const Point& point) {
    static_assert(D == 1, "Euclidean QCQP traits support D=1 only.");
    Matrix result(N + 1, 1);
    result(0, 0) = 1.0;
    result.template bottomRows<N>() = point;
    return result;
  }

  /** Fix the homogeneous scale; the physical coordinates are unconstrained. */
  template <int D = 1>
  static std::vector<std::pair<Matrix, double>> QcqpConstraints() {
    static_assert(D == 1, "Euclidean QCQP traits support D=1 only.");
    Matrix A = Matrix::Zero(N + 1, N + 1);
    A(0, 0) = 1.0;
    return {{A, 1.0}};
  }

  /** Decode a finite homogeneous column with nonzero scale. */
  template <int D = 1>
  static Point FromQcqpValue(const Matrix& value) {
    static_assert(D == 1, "Euclidean QCQP traits support D=1 only.");
    if (value.rows() != N + 1 || value.cols() != 1 ||
        !value.allFinite() || std::abs(value(0, 0)) < 1e-12) {
      throw std::invalid_argument(
          "Euclidean FromQcqpValue requires a finite homogeneous column "
          "of the expected dimension with nonzero scale.");
    }
    return value.template bottomRows<N>() / value(0, 0);
  }
};

}  // namespace gtsam
