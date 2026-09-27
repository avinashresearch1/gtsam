/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file testLiftedSDPs.cpp
 * @brief Tests for QCQP-backed lifted SDP objective assembly.
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/certifiable/LiftedSDPProblem.h>
#include <gtsam/constrained/QcqpProblem.h>
#include <gtsam/constrained/QpCost.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/FrobeniusFactor.h>
#include <gtsam/slam/RelativeTranslationFactor.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>

using namespace gtsam;

/* ************************************************************************* */
namespace lifted_sdp_tests {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Build a cycle of relative Rot2 measurements.
NonlinearFactorGraph Rot2RingGraph(size_t numPoses, double delta) {
  NonlinearFactorGraph graph;
  for (size_t i = 0; i < numPoses; ++i) {
    graph.emplace_shared<FrobeniusBetweenFactor<Rot2>>(
        Symbol('x', i), Symbol('x', (i + 1) % numPoses),
        Rot2::fromAngle(delta));
  }
  return graph;
}

// Create lifted QCQP values with a controlled angular perturbation.
Values Rot2RingQcqpValues(size_t numPoses, double delta, double perturbation) {
  Values values;
  for (size_t i = 0; i < numPoses; ++i) {
    InsertQcqpValue<Rot2, 1>(
        Symbol('x', i),
        Rot2::fromAngle(i * delta + perturbation * static_cast<double>(i)),
        &values);
  }
  return values;
}

// Build a cycle using relative measurements from the supplied poses.
template <typename T>
NonlinearFactorGraph PoseRingGraph(const std::vector<T>& poses) {
  NonlinearFactorGraph graph;
  for (size_t i = 0; i < poses.size(); ++i) {
    const size_t j = (i + 1) % poses.size();
    graph.emplace_shared<FrobeniusBetweenFactor<T>>(
        Symbol('x', i), Symbol('x', j), poses[i].between(poses[j]));
  }
  return graph;
}

// Lift a sequence of manifold values into D=1 QCQP coordinates.
template <typename T>
Values PoseRingQcqpValues(const std::vector<T>& poses) {
  Values values;
  for (size_t i = 0; i < poses.size(); ++i) {
    InsertQcqpValue<T, 1>(Symbol('x', i), poses[i], &values);
  }
  return values;
}

// Create a closed Pose2 ring by repeatedly composing one body-frame step.
std::vector<Pose2> Pose2RingPoses(size_t numPoses) {
  const Pose2 step(2.0, 0.0, 2.0 * kPi / static_cast<double>(numPoses));
  std::vector<Pose2> poses(numPoses);
  for (size_t i = 1; i < numPoses; ++i) {
    poses[i] = poses[i - 1].compose(step);
  }
  return poses;
}

// Create a closed Pose3 ring by repeatedly composing one body-frame step.
std::vector<Pose3> Pose3RingPoses(size_t numPoses) {
  const Pose3 step(Rot3::Rz(2.0 * kPi / static_cast<double>(numPoses)),
                   Point3(2.0, 0.0, 0.0));
  std::vector<Pose3> poses(numPoses);
  for (size_t i = 1; i < numPoses; ++i) {
    poses[i] = poses[i - 1].compose(step);
  }
  return poses;
}

// Apply deterministic local perturbations so the objective is nonzero.
std::vector<Pose2> PerturbedPose2Values(const std::vector<Pose2>& poses) {
  std::vector<Pose2> perturbed = poses;
  for (size_t i = 0; i < perturbed.size(); ++i) {
    const double scale = static_cast<double>(i);
    perturbed[i] =
        poses[i].retract(Vector3(0.01 * scale, -0.005 * scale, 0.003 * scale));
  }
  return perturbed;
}

// Apply deterministic local perturbations so the objective is nonzero.
std::vector<Pose3> PerturbedPose3Values(const std::vector<Pose3>& poses) {
  std::vector<Pose3> perturbed = poses;
  for (size_t i = 0; i < perturbed.size(); ++i) {
    const double scale = static_cast<double>(i);
    Vector6 delta{0.002 * scale, -0.001 * scale, 0.0015 * scale,
                  0.01 * scale,  -0.005 * scale, 0.004 * scale};
    perturbed[i] = poses[i].retract(delta);
  }
  return perturbed;
}

// Assemble the local rank-one matrix in a Hessian factor's key order.
Matrix BuildLocalX(const HessianFactor& H, const Values& values) {
  std::vector<Vector> localValues;
  DenseIndex totalDim = 0;
  for (auto it = H.begin(); it != H.end(); ++it) {
    Vector x = values.at<Matrix>(*it).col(0);
    if (x.size() != H.getDim(it)) {
      throw std::runtime_error("BuildLocalX: QCQP value dimension mismatch.");
    }
    totalDim += x.size();
    localValues.push_back(std::move(x));
  }

  Matrix X_f = Matrix::Zero(totalDim, totalDim);
  DenseIndex rowStart = 0;
  for (const Vector& x_i : localValues) {
    DenseIndex colStart = 0;
    for (const Vector& x_j : localValues) {
      X_f.block(rowStart, colStart, x_i.size(), x_j.size()) =
          x_i * x_j.transpose();
      colStart += x_j.size();
    }
    rowStart += x_i.size();
  }
  return X_f;
}

// Evaluate the lifted trace objective directly from rank-one matrices.
double ComputeLiftedObjective(const QcqpProblem& problem,
                              const Values& values) {
  double objective = 0.0;
  for (const auto& factor : problem.costs()) {
    if (!factor) {
      continue;
    }

    const auto* cost = dynamic_cast<const QpCost*>(factor.get());
    if (!cost) {
      throw std::runtime_error("ComputeLiftedObjective: expected QpCost.");
    }

    const HessianFactor& H = cost->hessianFactor();
    if (H.linearTerm().norm() > 0.0 || H.constantTerm() != 0.0) {
      throw std::runtime_error(
          "ComputeLiftedObjective: linear/constant terms are not supported.");
    }

    const Matrix Q_f = H.information();
    const Matrix X_f = BuildLocalX(H, values);
    objective += 0.5 * Q_f.cwiseProduct(X_f).sum();
  }
  return objective;
}

// Verifies that lifting preserves the QCQP objective at a feasible assignment.
TEST(LiftedSDPs, Rot2_QcqpObjectiveMatchesLiftedObjective) {
  constexpr size_t N = 20;
  const double delta = 2.0 * kPi / static_cast<double>(N);
  constexpr double perturbation = 0.03;

  const NonlinearFactorGraph graph = Rot2RingGraph(N, delta);
  const QcqpProblem problem(graph);
  const Values qcqpValues = Rot2RingQcqpValues(N, delta, perturbation);

  const double qcqpObjective = problem.costs().error(qcqpValues);
  const double sdpObjective = ComputeLiftedObjective(problem, qcqpValues);

  std::cout << "qcqpObjective: " << qcqpObjective << std::endl;
  std::cout << "sdpObjective: " << sdpObjective << std::endl;

  EXPECT_DOUBLES_EQUAL(qcqpObjective, sdpObjective, 1e-12);
}

// Verifies that the Pose2 QCQP and its rank-one SDP lift have equal costs.
TEST(LiftedSDPs, Pose2_QcqpObjectiveMatchesLiftedObjective) {
  constexpr size_t N = 20;
  const std::vector<Pose2> groundTruth = Pose2RingPoses(N);
  const NonlinearFactorGraph graph = PoseRingGraph(groundTruth);
  const QcqpProblem problem(graph);
  const Values qcqpValues =
      PoseRingQcqpValues(PerturbedPose2Values(groundTruth));

  const double qcqpObjective = problem.costs().error(qcqpValues);
  const double sdpObjective = ComputeLiftedObjective(problem, qcqpValues);

  EXPECT_DOUBLES_EQUAL(qcqpObjective, sdpObjective, 1e-10);
}

// Verifies that the Pose3 QCQP and its rank-one SDP lift have equal costs.
TEST(LiftedSDPs, Pose3_QcqpObjectiveMatchesLiftedObjective) {
  constexpr size_t N = 20;
  const std::vector<Pose3> groundTruth = Pose3RingPoses(N);
  const NonlinearFactorGraph graph = PoseRingGraph(groundTruth);
  const QcqpProblem problem(graph);
  const Values qcqpValues =
      PoseRingQcqpValues(PerturbedPose3Values(groundTruth));

  const double qcqpObjective = problem.costs().error(qcqpValues);
  const double sdpObjective = ComputeLiftedObjective(problem, qcqpValues);

  EXPECT_DOUBLES_EQUAL(qcqpObjective, sdpObjective, 1e-10);
}

}  // namespace lifted_sdp_tests
/* ************************************************************************* */

#ifdef GTSAM_USE_MOSEK
/* ************************************************************************* */
namespace pose_ring_sdp_fixture {

constexpr size_t kNumPoses = 20;
constexpr double kRankOneEigenRatioThreshold = 1e5;
constexpr double kPoseErrorTolerance = 1e-4;
constexpr double kObjectiveTolerance = 1e-3;

struct SdpSolutionSummary {
  double objective = 0.0;
  double minimumEigenvalueRatio = 0.0;
  double maximumPoseError = 0.0;
  bool finiteEigenvalueRatios = false;
  bool repeatedQueriesMatch = false;
};

// Build an exactly consistent ring with a hard Frobenius prior on the first
// pose.
template <typename T>
NonlinearFactorGraph ExactPoseRingGraph(const std::vector<T>& poses,
                                        size_t frobeniusDimension) {
  NonlinearFactorGraph graph;
  const auto priorNoise = noiseModel::Constrained::All(frobeniusDimension);
  const auto betweenNoise = noiseModel::Isotropic::Sigma(T::dimension, 0.01);
  graph.emplace_shared<FrobeniusPrior<T>>(0, poses[0].matrix(), priorNoise);
  for (size_t i = 0; i < poses.size(); ++i) {
    const size_t j = (i + 1) % poses.size();
    graph.emplace_shared<FrobeniusBetweenFactor<T>>(
        i, j, poses[i].between(poses[j]), betweenNoise);
  }
  return graph;
}

// Solve an SDP and summarize its diagonal-block rank and recovery accuracy.
template <typename T, typename SdpProblem>
SdpSolutionSummary SolveAndSummarize(SdpProblem* sdp,
                                     const std::vector<T>& groundTruth) {
  const std::map<std::string, double> mosekParams{
      {"intpntCoTolRelGap", 1e-10},
      {"optimizerMaxTime", 1500.0},
  };
  if (!sdp->solve(mosekParams)) {
    throw std::runtime_error("MOSEK did not return a readable solution.");
  }

  const std::vector<double> eigenvalueRatios = sdp->variableEVRs();
  const Values qcqpValues = sdp->qcqpValues();
  const Values repeatedQcqpValues = sdp->qcqpValues();
  const std::vector<double> repeatedEigenvalueRatios = sdp->variableEVRs();
  const bool finiteEigenvalueRatios =
      std::all_of(eigenvalueRatios.begin(), eigenvalueRatios.end(),
                  [](double ratio) { return std::isfinite(ratio); });
  const bool repeatedRatiosMatch =
      eigenvalueRatios.size() == repeatedEigenvalueRatios.size() &&
      std::equal(eigenvalueRatios.begin(), eigenvalueRatios.end(),
                 repeatedEigenvalueRatios.begin(),
                 [](double first, double second) {
                   return std::abs(first - second) <= 1e-12;
                 });
  const bool repeatedQueriesMatch =
      repeatedRatiosMatch &&
      assert_equal(qcqpValues, repeatedQcqpValues, 1e-12);

  const auto recovered = ExtractQcqpValues<T, 1>(qcqpValues);
  if (recovered.size() != groundTruth.size()) {
    throw std::runtime_error(
        "Recovered QCQP value count does not match ground truth.");
  }
  std::vector<double> poseErrors(recovered.size());
  for (size_t index = 0; index < recovered.size(); ++index) {
    poseErrors[index] =
        groundTruth[index].localCoordinates(recovered[index].second).norm();
  }

  return {sdp->objectiveValue(),
          *std::min_element(eigenvalueRatios.begin(), eigenvalueRatios.end()),
          *std::max_element(poseErrors.begin(), poseErrors.end()),
          finiteEigenvalueRatios, repeatedQueriesMatch};
}

// Recovery queries reject access before either SDP formulation has been solved.
TEST(LiftedSDPs, RecoveryQueriesRequireSolve) {
  const std::vector<Pose2> groundTruth =
      lifted_sdp_tests::Pose2RingPoses(kNumPoses);
  const QcqpProblem problem(ExactPoseRingGraph(groundTruth, 9));
  LiftedSDPProblem<MonolithicSDP, MosekSDPSolver> monolithic(problem);
  LiftedSDPProblem<ChordalSDP, MosekSDPSolver> chordal(
      problem, ChordalOrderingType::Metis);

  CHECK_EXCEPTION(monolithic.qcqpValues(), std::runtime_error);
  CHECK_EXCEPTION(monolithic.variableEVRs(), std::runtime_error);
  CHECK_EXCEPTION(chordal.qcqpValues(), std::runtime_error);
  CHECK_EXCEPTION(chordal.variableEVRs(), std::runtime_error);
}

// Verifies rank-one Pose2 slices and matching monolithic/chordal solutions.
TEST(LiftedSDPs, Pose2_MonolithicAndChordal) {
  const std::vector<Pose2> groundTruth =
      lifted_sdp_tests::Pose2RingPoses(kNumPoses);
  const QcqpProblem problem(ExactPoseRingGraph(groundTruth, 9));

  LiftedSDPProblem<MonolithicSDP, MosekSDPSolver> monolithic(problem);
  LiftedSDPProblem<ChordalSDP, MosekSDPSolver> chordal(
      problem, ChordalOrderingType::Metis);
  const SdpSolutionSummary monolithicResult =
      SolveAndSummarize(&monolithic, groundTruth);
  const SdpSolutionSummary chordalResult =
      SolveAndSummarize(&chordal, groundTruth);

  EXPECT(monolithicResult.minimumEigenvalueRatio > kRankOneEigenRatioThreshold);
  EXPECT(chordalResult.minimumEigenvalueRatio > kRankOneEigenRatioThreshold);
  EXPECT(monolithicResult.finiteEigenvalueRatios);
  EXPECT(chordalResult.finiteEigenvalueRatios);
  EXPECT(monolithicResult.repeatedQueriesMatch);
  EXPECT(chordalResult.repeatedQueriesMatch);
  EXPECT(monolithicResult.maximumPoseError < kPoseErrorTolerance);
  EXPECT(chordalResult.maximumPoseError < kPoseErrorTolerance);
  EXPECT(monolithicResult.objective < kObjectiveTolerance);
  EXPECT(chordalResult.objective < kObjectiveTolerance);
  EXPECT_DOUBLES_EQUAL(monolithicResult.objective, chordalResult.objective,
                       kObjectiveTolerance);
}

// Verifies rank-one Pose3 slices and matching monolithic/chordal solutions.
TEST(LiftedSDPs, Pose3_MonolithicAndChordal) {
  const std::vector<Pose3> groundTruth =
      lifted_sdp_tests::Pose3RingPoses(kNumPoses);
  const QcqpProblem problem(ExactPoseRingGraph(groundTruth, 16));

  LiftedSDPProblem<MonolithicSDP, MosekSDPSolver> monolithic(problem);
  LiftedSDPProblem<ChordalSDP, MosekSDPSolver> chordal(
      problem, ChordalOrderingType::Metis);
  const SdpSolutionSummary monolithicResult =
      SolveAndSummarize(&monolithic, groundTruth);
  const SdpSolutionSummary chordalResult =
      SolveAndSummarize(&chordal, groundTruth);

  EXPECT(monolithicResult.minimumEigenvalueRatio > kRankOneEigenRatioThreshold);
  EXPECT(chordalResult.minimumEigenvalueRatio > kRankOneEigenRatioThreshold);
  EXPECT(monolithicResult.finiteEigenvalueRatios);
  EXPECT(chordalResult.finiteEigenvalueRatios);
  EXPECT(monolithicResult.repeatedQueriesMatch);
  EXPECT(chordalResult.repeatedQueriesMatch);
  EXPECT(monolithicResult.maximumPoseError < kPoseErrorTolerance);
  EXPECT(chordalResult.maximumPoseError < kPoseErrorTolerance);
  EXPECT(monolithicResult.objective < kObjectiveTolerance);
  EXPECT(chordalResult.objective < kObjectiveTolerance);
  EXPECT_DOUBLES_EQUAL(monolithicResult.objective, chordalResult.objective,
                       kObjectiveTolerance);
}


// Separator blocks repeated across several levels retain consistent recovery.
TEST(LiftedSDPs, ChordalTreeOverlapRecovery) {
  std::vector<Pose2> truth;
  for (size_t i = 0; i < 24; ++i) {
    const double angle = 2 * lifted_sdp_tests::kPi * i / 24;
    truth.emplace_back(std::cos(angle), std::sin(angle), angle);
  }
  const Pose2 first = truth.front();
  for (auto& pose : truth) pose = first.between(pose);
  const QcqpProblem problem(ExactPoseRingGraph(truth, 9));
  for (auto ordering : {ChordalOrderingType::Metis,
                        ChordalOrderingType::Colamd}) {
    MosekChordalSDP chordal(problem, ordering);
    size_t maximumDepth = 0;
    std::function<void(const SymbolicBayesTree::sharedClique&, size_t)> visit;
    visit = [&](const SymbolicBayesTree::sharedClique& clique, size_t depth) {
      maximumDepth = std::max(maximumDepth, depth);
      for (const auto& child : clique->children) visit(child, depth + 1);
    };
    for (const auto& root : chordal.bayesTree().roots()) visit(root, 1);
    EXPECT(maximumDepth >= 3);
    const auto result = SolveAndSummarize(&chordal, truth);
    EXPECT(std::abs(result.objective) < kObjectiveTolerance);
    EXPECT(result.maximumPoseError < kPoseErrorTolerance);
    EXPECT(result.minimumEigenvalueRatio > kRankOneEigenRatioThreshold);
  }
}

}  // namespace pose_ring_sdp_fixture
/* ************************************************************************* */
namespace split_ring_sdp_tests {

// Mixed rotation/translation blocks recover the same exact anchored spatial ring.
TEST(LiftedSDPs, SplitPose3Ring) {
  using symbol_shorthand::R;
  using symbol_shorthand::T;
  std::vector<Pose3> truth;
  for (size_t i = 0; i < 5; ++i) {
    const double theta = 2 * lifted_sdp_tests::kPi * i / 5;
    truth.emplace_back(Rot3::RzRyRx(0.2 * std::sin(theta), 0.1 * std::cos(theta), theta),
                       Point3(std::cos(theta), std::sin(theta), 0.3 * std::sin(2 * theta)));
  }
  const Pose3 first = truth.front();
  for (auto& pose : truth) pose = first.between(pose);
  NonlinearFactorGraph graph;
  for (size_t i = 0; i < truth.size(); ++i) {
    const size_t j = (i + 1) % truth.size();
    const Pose3 relative = truth[i].between(truth[j]);
    graph.emplace_shared<FrobeniusBetweenFactor<Rot3>>(
        R(i), R(j), relative.rotation(), noiseModel::Isotropic::Variance(3, 0.1));
    graph.emplace_shared<RelativeTranslationFactor3>(
        R(i), T(i), T(j), relative.translation(), 2.5);
  }
  QcqpProblem problem(graph);
  problem.fixValue(R(0), qcqpValue(Rot3()));
  problem.fixValue(T(0), qcqpValue(Point3(0, 0, 0)));
  MosekMonolithicSDP monolithic(problem);
  MosekChordalSDP chordal(problem, ChordalOrderingType::Metis);
  auto check = [&](auto& solver) {
    EXPECT(solver.solve());
    EXPECT(std::abs(solver.objectiveValue()) < 1e-6);
    const Values recovered = solver.qcqpValues();
    const Values rotations = extractQcqpValues<Rot3>(recovered);
    const Values points = extractQcqpValues<Point3>(recovered);
    Values estimate;
    for (size_t i = 0; i < truth.size(); ++i) {
      estimate.insert(R(i), rotations.at<Rot3>(R(i)));
      estimate.insert(T(i), points.at<Point3>(T(i)));
      EXPECT(assert_equal(truth[i].rotation(), estimate.at<Rot3>(R(i)), 1e-4));
      EXPECT(assert_equal(truth[i].translation(), estimate.at<Point3>(T(i)), 1e-4));
    }
    EXPECT(graph.error(estimate) < 1e-6);
    for (double ratio : solver.variableEVRs()) EXPECT(ratio > 1e5);
  };
  check(monolithic);
  check(chordal);
  EXPECT_LONGS_EQUAL(10, graph.size());
  EXPECT_DOUBLES_EQUAL(monolithic.objectiveValue(), chordal.objectiveValue(), 1e-6);
}


// Substitution preserves nonzero fixed values and costs involving only them.
TEST(LiftedSDPs, FixedValuesAndConstantCost) {
  QcqpProblem problem;
  const Vector4 residual{-1, -1, 0, 1};
  const Vector4 fixedResidual{0, 1, 0, -1};
  problem.addCost(QpCost({0, 1}, SymmetricBlockMatrix(
      std::vector<DenseIndex>{2, 2}, Matrix(residual * residual.transpose()))));
  problem.addCost(QpCost({0, 2}, SymmetricBlockMatrix(
      std::vector<DenseIndex>{2, 2}, Matrix(fixedResidual * fixedResidual.transpose()))));
  Matrix2 normalization = Matrix2::Zero();
  normalization(0, 0) = 1;
  for (Key key : {0, 1, 2})
    problem.addConstraint(QuadraticConstraint::Equal(key, normalization, 1));
  const Matrix first = Vector2{1, 2}, last = Vector2{1, -1};
  problem.fixValue(0, first);
  problem.fixValue(2, last);
  MosekChordalSDP chordal(problem, ChordalOrderingType::Metis);
  EXPECT(chordal.solve());
  EXPECT_DOUBLES_EQUAL(4.5, chordal.objectiveValue(), 1e-7);
  const Values result = chordal.qcqpValues();
  EXPECT_LONGS_EQUAL(3, result.size());
  EXPECT(assert_equal(first, result.at<Matrix>(0), 0.0));
  EXPECT(assert_equal(last, result.at<Matrix>(2), 0.0));
  EXPECT(assert_equal(Matrix(Vector2{1, 3}), result.at<Matrix>(1), 1e-6));
  EXPECT_LONGS_EQUAL(3, chordal.orderedKeys().size());
  for (double ratio : chordal.variableEVRs()) EXPECT(std::isfinite(ratio));
}

}  // namespace split_ring_sdp_tests
/* ************************************************************************* */
#endif

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
