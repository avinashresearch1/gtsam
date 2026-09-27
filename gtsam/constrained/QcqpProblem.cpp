/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    QcqpProblem.cpp
 * @brief   QCQP problem implementations.
 * @author  Frank Dellaert
 */

#include <gtsam/constrained/QcqpProblem.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace gtsam {
namespace {

using QuadraticConstraintIndex =
    std::unordered_map<Key, std::vector<QuadraticConstraint>>;

/* ************************************************************************* */
bool SameQuadraticEquality(const QuadraticConstraint& first,
                           const QuadraticConstraint& second) {
  return first.key() == second.key() && first.A().isApprox(second.A(), 0.0) &&
         first.b() == second.b() && first.sigma() == second.sigma();
}

/* ************************************************************************* */
void MergeEqualityConstraints(
    const NonlinearEqualityConstraints& source,
    NonlinearEqualityConstraints* destination,
    QuadraticConstraintIndex* quadraticConstraintIndex) {
  for (const auto& factor : source) {
    const auto quadratic =
        std::dynamic_pointer_cast<QuadraticEqualityConstraintFactor>(factor);
    if (!quadratic) {
      destination->push_back(factor);
      continue;
    }

    const QuadraticConstraint& constraint = quadratic->quadraticConstraint();
    auto& sameKeyConstraints = (*quadraticConstraintIndex)[constraint.key()];
    const bool alreadyPresent =
        std::any_of(sameKeyConstraints.begin(), sameKeyConstraints.end(),
                    [&constraint](const QuadraticConstraint& existing) {
                      return SameQuadraticEquality(existing, constraint);
                    });
    if (!alreadyPresent) {
      destination->push_back(factor);
      sameKeyConstraints.push_back(constraint);
    }
  }
}

}  // namespace

/* ************************************************************************* */
QcqpProblem::QcqpProblem(const NonlinearFactorGraph& graph,
                         size_t columnDimension) {
  if (columnDimension == 0) {
    throw std::invalid_argument(
        "QcqpProblem: columnDimension must be positive.");
  }

  QuadraticConstraintIndex quadraticConstraintIndex;
  for (const auto& factor : graph) {
    if (!factor) continue;

    NonlinearFactorGraph factorCosts;
    NonlinearEqualityConstraints factorConstraints;
    factor->qcqpFactors(&factorCosts, &factorConstraints, columnDimension);
    costs_.add(factorCosts);
    MergeEqualityConstraints(factorConstraints, &eqConstraints_,
                             &quadraticConstraintIndex);
  }
}

/* ************************************************************************* */
void QcqpProblem::addConstraint(const LinearConstraint& constraint) {
  if (constraint.isEquality()) {
    eqConstraints_.push_back(constraint.createEqualityFactor());
  } else {
    ineqConstraints_.push_back(constraint.createInequalityFactor());
  }
}

/* ************************************************************************* */
void QcqpProblem::fixValue(Key key, const Matrix& value) {
  if (value.cols() != 1 || value.rows() < 2 || !value.allFinite() ||
      std::abs(value(0, 0) - 1.0) > 1e-12) {
    throw std::invalid_argument(
        "QcqpProblem::fixValue requires a finite homogeneous column [1; value].");
  }
  bool found = false;
  for (const auto& factor : costs_) {
    const auto cost = std::dynamic_pointer_cast<QpCost>(factor);
    if (!cost) continue;
    const auto& hessian = cost->hessianFactor();
    for (auto it = hessian.begin(); it != hessian.end(); ++it) {
      if (*it != key) continue;
      found = true;
      if (hessian.getDim(it) != value.rows()) {
        throw std::invalid_argument("QcqpProblem::fixValue dimension mismatch.");
      }
    }
  }
  if (!found) {
    throw std::invalid_argument("QcqpProblem::fixValue key is not in the objective.");
  }
  addConstraint(LinearConstraint::Equal(
      JacobianFactor(key, Matrix::Identity(value.rows(), value.rows()),
                     Vector(value.col(0)))));
}

/* ************************************************************************* */
void QcqpProblem::addConstraint(const QuadraticConstraint& constraint) {
  if (constraint.isEquality()) {
    eqConstraints_.push_back(constraint.createEqualityFactor());
  } else {
    ineqConstraints_.push_back(constraint.createInequalityFactor());
  }
}

}  // namespace gtsam
