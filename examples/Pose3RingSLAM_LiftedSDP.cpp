/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Pose3RingSLAM_LiftedSDP.cpp
 * @brief Benchmark Pose3 ring SLAM with lifted SDP and local solvers.
 */

#include <gtsam/certifiable/LiftedSDPProblem.h>
#include <gtsam/constrained/QcqpProblem.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/Sampler.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/FrobeniusFactor.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gtsam;

template <typename T>
std::vector<T> createPoses(const T& initialPose, size_t N, const T& step) {
  std::vector<T> poses;
  poses.reserve(N);
  poses.push_back(initialPose);
  for (size_t i = 1; i < N; ++i) {
    poses.push_back(poses.back().compose(step));
  }
  return poses;
}

namespace {

constexpr double kDefaultOdometryNoise = 0.01;
constexpr uint32_t kDefaultSeed = 42u;
constexpr double kMaxOptimizerTimeSeconds = 1500.0;
constexpr double kRankOneEigenRatioThreshold = 1e5;
constexpr size_t kPose3FrobeniusDimension = 16;
constexpr int kDefaultMosekNumThreads = 1;

enum class SolverMode { Monolithic, Chordal, LocalRandom, LocalGroundTruth };

struct CommandLineOptions {
  SolverMode solverMode = SolverMode::Monolithic;
  std::string solver;
  size_t N = 20;
  double odometryNoise = kDefaultOdometryNoise;
  double sampleNoise = -1.0;
  uint32_t seed = kDefaultSeed;
  uint32_t initSeed = kDefaultSeed;
  int mosekNumThreads = kDefaultMosekNumThreads;
  ChordalOrderingType ordering = ChordalOrderingType::Metis;
  bool quiet = false;
};

struct RingProblem {
  NonlinearFactorGraph graph;
  std::vector<Pose3> groundTruth;
};

struct PoseErrorSummary {
  double minimum;
  double firstQuartile;
  double median;
  double thirdQuartile;
  double maximum;
};

struct BenchmarkResult {
  std::string solver;
  size_t N;
  double odometryNoise;
  double sampleNoise;
  uint32_t seed;
  double qcqpBuildSeconds = 0.0;
  double sdpBuildSeconds = 0.0;
  double solveTimeSeconds = 0.0;
  double solveWallSeconds = 0.0;
  double recoverSeconds = 0.0;
  double averagePoseErrorNorm = 0.0;
  PoseErrorSummary poseErrorSummary{};
  std::optional<std::string> problemStatus;
  std::optional<double> objectiveValue;
  std::optional<size_t> rankOnePoseCount;
  std::optional<size_t> rankOnePoseTotal;
  std::optional<double> minEVR;
  std::optional<double> maxEVR;
  std::optional<double> averageEVR;
  std::optional<int> mosekNumThreads;
  std::optional<size_t> maxCliqueSize;
  std::optional<size_t> numCliques;
};

SolverMode parseSolver(const std::string& value) {
  if (value == "monolithic") return SolverMode::Monolithic;
  if (value == "chordal") return SolverMode::Chordal;
  if (value == "local-random") return SolverMode::LocalRandom;
  if (value == "local-gt") return SolverMode::LocalGroundTruth;
  throw std::runtime_error("Unknown solver: " + value);
}

ChordalOrderingType parseOrdering(const std::string& value) {
  if (value == "metis") return ChordalOrderingType::Metis;
  if (value == "colamd") return ChordalOrderingType::Colamd;
  throw std::runtime_error("Unknown ordering: " + value);
}

CommandLineOptions parseCommandLine(int argc, char** argv) {
  CommandLineOptions options;
  bool hasSolver = false;

  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    auto requireValue = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value.");
      }
      return argv[++i];
    };

    if (argument == "--solver") {
      options.solver = requireValue(argument);
      options.solverMode = parseSolver(options.solver);
      hasSolver = true;
    } else if (argument == "--N") {
      options.N = std::stoul(requireValue(argument));
    } else if (argument == "--odometry-noise") {
      options.odometryNoise = std::stod(requireValue(argument));
    } else if (argument == "--sample-noise") {
      options.sampleNoise = std::stod(requireValue(argument));
    } else if (argument == "--seed") {
      options.seed = static_cast<uint32_t>(std::stoul(requireValue(argument)));
    } else if (argument == "--init-seed") {
      options.initSeed =
          static_cast<uint32_t>(std::stoul(requireValue(argument)));
    } else if (argument == "--mosek-threads") {
      options.mosekNumThreads = std::stoi(requireValue(argument));
    } else if (argument == "--ordering") {
      options.ordering = parseOrdering(requireValue(argument));
    } else if (argument == "--quiet") {
      options.quiet = true;
    } else {
      throw std::runtime_error("Unknown argument: " + argument);
    }
  }

  if (!hasSolver) throw std::runtime_error("--solver must be specified.");
  if (options.N == 0) throw std::runtime_error("--N must be positive.");
  if (options.odometryNoise <= 0.0) {
    throw std::runtime_error("--odometry-noise must be positive.");
  }
  if (options.sampleNoise < 0.0) {
    options.sampleNoise = options.odometryNoise;
  }
  if (options.sampleNoise <= 0.0) {
    throw std::runtime_error("--sample-noise must be positive.");
  }
  if (options.mosekNumThreads <= 0) {
    throw std::runtime_error("--mosek-threads must be positive.");
  }
  return options;
}

RingProblem buildRingProblem(const CommandLineOptions& options) {
  RingProblem problem;

  // Match the original edge length at N=100, then hold that radius fixed as N
  // changes so the benchmark varies discretization rather than problem scale.
  const double referenceYawStep = 2.0 * M_PI / 100.0;
  const double fixedRadius = 2.0 / (2.0 * std::sin(referenceYawStep / 2.0));
  const double yawStep = 2.0 * M_PI / static_cast<double>(options.N);
  const double translationStep =
      2.0 * fixedRadius * std::sin(yawStep / 2.0);
  const Pose3 step(Rot3::Rz(yawStep), Point3(translationStep, 0.0, 0.0));
  problem.groundTruth = createPoses(Pose3(), options.N, step);

  auto odometryNoiseModel =
      noiseModel::Isotropic::Sigma(Pose3::dimension, options.odometryNoise);
  auto sampledNoiseModel =
      noiseModel::Isotropic::Sigma(Pose3::dimension, options.sampleNoise);
  Sampler odometrySampler(sampledNoiseModel, options.seed);

  auto exactPriorNoiseModel =
      noiseModel::Constrained::All(kPose3FrobeniusDimension);
  problem.graph.emplace_shared<FrobeniusPrior<Pose3>>(
      0, problem.groundTruth[0].matrix(), exactPriorNoiseModel);

  for (size_t i = 0; i < options.N; ++i) {
    const size_t j = (i + 1) % options.N;
    const Pose3 measurement =
        problem.groundTruth[i].between(problem.groundTruth[j]);
    const Pose3 noisyMeasurement =
        measurement.retract(odometrySampler.sample());
    problem.graph.emplace_shared<FrobeniusBetweenFactor<Pose3>>(
        i, j, noisyMeasurement, odometryNoiseModel);
  }
  return problem;
}

Values makeGroundTruthInitialEstimate(const std::vector<Pose3>& groundTruth) {
  Values initialEstimate;
  for (size_t i = 0; i < groundTruth.size(); ++i) {
    initialEstimate.insert(i, groundTruth[i]);
  }
  return initialEstimate;
}

Values makeRandomInitialEstimate(size_t N, uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> translation(-2.0, 2.0);

  Values initialEstimate;
  for (size_t i = 0; i < N; ++i) {
    initialEstimate.insert(
        i, Pose3(Rot3::Random(generator),
                 Point3(translation(generator), translation(generator),
                        translation(generator))));
  }
  return initialEstimate;
}

double interpolateQuantile(const std::vector<double>& sortedValues,
                           double quantile) {
  const double position =
      quantile * static_cast<double>(sortedValues.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return (1.0 - fraction) * sortedValues[lower] +
         fraction * sortedValues[upper];
}

PoseErrorSummary summarizePoseErrors(std::vector<double> errors) {
  if (errors.empty()) throw std::runtime_error("No pose errors to summarize.");
  std::sort(errors.begin(), errors.end());
  return {errors.front(), interpolateQuantile(errors, 0.25),
          interpolateQuantile(errors, 0.5),
          interpolateQuantile(errors, 0.75), errors.back()};
}

double average(const std::vector<double>& values) {
  if (values.empty()) throw std::runtime_error("No values to average.");
  double total = 0.0;
  for (double value : values) total += value;
  return total / static_cast<double>(values.size());
}

std::vector<double> poseErrors(
    const std::vector<std::pair<Key, Pose3>>& estimates,
    const std::vector<Pose3>& groundTruth) {
  if (estimates.size() != groundTruth.size()) {
    throw std::runtime_error(
        "Recovered QCQP value count does not match ground truth.");
  }

  std::vector<double> errors;
  errors.reserve(estimates.size());
  for (const auto& [key, estimate] : estimates) {
    if (key >= groundTruth.size()) {
      throw std::runtime_error("Recovered QCQP key is outside ground truth.");
    }
    errors.push_back(
        Pose3::Logmap(groundTruth[key].between(estimate)).norm());
  }
  return errors;
}

std::vector<double> poseErrors(const Values& estimates,
                               const std::vector<Pose3>& groundTruth) {
  std::vector<double> errors;
  errors.reserve(groundTruth.size());
  for (size_t i = 0; i < groundTruth.size(); ++i) {
    errors.push_back(Pose3::Logmap(
                         groundTruth[i].between(estimates.at<Pose3>(i)))
                         .norm());
  }
  return errors;
}

void printPoses(const std::string& solver,
                const std::vector<std::pair<Key, Pose3>>& estimates,
                const std::vector<Pose3>& groundTruth) {
  for (const auto& [key, estimate] : estimates) {
    groundTruth.at(key).print(solver + " ground-truth pose " +
                              std::to_string(key));
    estimate.print(solver + " recovered pose " + std::to_string(key));
  }
}

void printPoses(const std::string& solver, const Values& estimates,
                const std::vector<Pose3>& groundTruth) {
  for (size_t i = 0; i < groundTruth.size(); ++i) {
    groundTruth[i].print(solver + " ground-truth pose " + std::to_string(i));
    estimates.at<Pose3>(i).print(solver + " recovered pose " +
                                 std::to_string(i));
  }
}

void printBenchmarkResult(const BenchmarkResult& result) {
  std::cout << std::setprecision(17)
            << "BENCHMARK_RESULT"
            << " solver=" << result.solver << " N=" << result.N
            << " odometry_noise=" << result.odometryNoise
            << " sample_noise=" << result.sampleNoise
            << " seed=" << result.seed
            << " solve_time_seconds=" << result.solveTimeSeconds
            << " solve_wall_seconds=" << result.solveWallSeconds
            << " qcqp_build_seconds=" << result.qcqpBuildSeconds
            << " sdp_build_seconds=" << result.sdpBuildSeconds
            << " recover_seconds=" << result.recoverSeconds
            << " average_pose_error_norm=" << result.averagePoseErrorNorm
            << " pose_error_min=" << result.poseErrorSummary.minimum
            << " pose_error_q1=" << result.poseErrorSummary.firstQuartile
            << " pose_error_median=" << result.poseErrorSummary.median
            << " pose_error_q3=" << result.poseErrorSummary.thirdQuartile
            << " pose_error_max=" << result.poseErrorSummary.maximum;
  if (result.problemStatus) {
    std::cout << " problem_status=" << *result.problemStatus;
  }
  if (result.objectiveValue) {
    std::cout << " objective_value=" << *result.objectiveValue;
  }
  if (result.rankOnePoseCount) {
    std::cout << " rank_one_pose_count=" << *result.rankOnePoseCount;
  }
  if (result.rankOnePoseTotal) {
    std::cout << " rank_one_pose_total=" << *result.rankOnePoseTotal;
  }
  if (result.minEVR) std::cout << " minEVR=" << *result.minEVR;
  if (result.maxEVR) std::cout << " maxEVR=" << *result.maxEVR;
  if (result.averageEVR) std::cout << " averageEVR=" << *result.averageEVR;
  if (result.mosekNumThreads) {
    std::cout << " mosek_num_threads=" << *result.mosekNumThreads;
  }
  if (result.maxCliqueSize) {
    std::cout << " max_clique_size=" << *result.maxCliqueSize;
  }
  if (result.numCliques) {
    std::cout << " num_cliques=" << *result.numCliques;
  }
  std::cout << std::endl;
}

template <typename SdpProblem>
BenchmarkResult solveSdp(const CommandLineOptions& options,
                         const std::vector<Pose3>& groundTruth,
                         SdpProblem* sdp, double qcqpBuildSeconds,
                         double sdpBuildSeconds) {
  const std::map<std::string, double> mosekParams{
      {"intpntCoTolRelGap", 1e-10},
      {"optimizerMaxTime", kMaxOptimizerTimeSeconds},
  };
  const std::map<std::string, int> integerMosekParams{
      {"numThreads", options.mosekNumThreads}};

  const auto solveStart = std::chrono::steady_clock::now();
  if (!sdp->solve(mosekParams, integerMosekParams)) {
    throw std::runtime_error(
        "MOSEK solve did not produce a readable primal solution.");
  }
  const double solveWallSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    solveStart)
          .count();
  const int mosekNumThreads = sdp->solveNumThreads();
  if (mosekNumThreads != options.mosekNumThreads) {
    throw std::runtime_error("MOSEK did not use the requested thread count.");
  }

  const auto recoverStart = std::chrono::steady_clock::now();
  const Values qcqpValues = sdp->qcqpValues();
  const auto estimates = ExtractQcqpValues<Pose3, 1>(qcqpValues);
  const double recoverSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    recoverStart)
          .count();
  const std::vector<double> errors = poseErrors(estimates, groundTruth);
  const std::vector<double> eigenvalueRatios = sdp->variableEVRs();

  if (!options.quiet) {
    printPoses(options.solver, estimates, groundTruth);
  }

  BenchmarkResult result;
  result.solver = options.solver;
  result.N = options.N;
  result.odometryNoise = options.odometryNoise;
  result.sampleNoise = options.sampleNoise;
  result.seed = options.seed;
  result.qcqpBuildSeconds = qcqpBuildSeconds;
  result.sdpBuildSeconds = sdpBuildSeconds;
  result.solveTimeSeconds = sdp->solveTimeSeconds();
  result.solveWallSeconds = solveWallSeconds;
  result.recoverSeconds = recoverSeconds;
  result.averagePoseErrorNorm = average(errors);
  result.poseErrorSummary = summarizePoseErrors(errors);
  result.problemStatus = sdp->problemStatus();
  result.objectiveValue = sdp->objectiveValue();
  result.rankOnePoseCount = static_cast<size_t>(std::count_if(
      eigenvalueRatios.begin(), eigenvalueRatios.end(), [](double ratio) {
        return ratio >= kRankOneEigenRatioThreshold;
      }));
  result.rankOnePoseTotal = eigenvalueRatios.size();
  result.minEVR =
      *std::min_element(eigenvalueRatios.begin(), eigenvalueRatios.end());
  result.maxEVR =
      *std::max_element(eigenvalueRatios.begin(), eigenvalueRatios.end());
  result.averageEVR = average(eigenvalueRatios);
  result.mosekNumThreads = mosekNumThreads;
  return result;
}

BenchmarkResult solveLocal(const CommandLineOptions& options,
                           const RingProblem& problem,
                           const Values& initialEstimate) {
  LevenbergMarquardtParams params;
  params.verbosity = NonlinearOptimizerParams::SILENT;
  params.verbosityLM = LevenbergMarquardtParams::SILENT;
  LevenbergMarquardtOptimizer optimizer(problem.graph, initialEstimate,
                                        params);

  const auto solveStart = std::chrono::steady_clock::now();
  const Values estimates = optimizer.optimize();
  const double solveSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    solveStart)
          .count();
  const std::vector<double> errors =
      poseErrors(estimates, problem.groundTruth);

  if (!options.quiet) {
    printPoses(options.solver, estimates, problem.groundTruth);
  }

  BenchmarkResult result;
  result.solver = options.solver;
  result.N = options.N;
  result.odometryNoise = options.odometryNoise;
  result.sampleNoise = options.sampleNoise;
  result.seed = options.seed;
  result.solveTimeSeconds = solveSeconds;
  result.solveWallSeconds = solveSeconds;
  result.averagePoseErrorNorm = average(errors);
  result.poseErrorSummary = summarizePoseErrors(errors);
  result.objectiveValue = problem.graph.error(estimates);
  return result;
}

std::pair<size_t, size_t> chordalCliqueMetrics(
    const SymbolicBayesTree& bayesTree) {
  const BayesTreeCliqueData cliqueData = bayesTree.getCliqueData();
  size_t maxCliqueSize = 0;
  for (size_t i = 0; i < cliqueData.conditionalSizes.size(); ++i) {
    maxCliqueSize =
        std::max(maxCliqueSize, cliqueData.conditionalSizes[i] +
                                    cliqueData.separatorSizes[i]);
  }
  return {maxCliqueSize, bayesTree.size()};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CommandLineOptions options = parseCommandLine(argc, argv);
    const RingProblem ringProblem = buildRingProblem(options);

    if (options.solverMode == SolverMode::LocalGroundTruth) {
      printBenchmarkResult(solveLocal(
          options, ringProblem,
          makeGroundTruthInitialEstimate(ringProblem.groundTruth)));
      return 0;
    }
    if (options.solverMode == SolverMode::LocalRandom) {
      printBenchmarkResult(solveLocal(
          options, ringProblem,
          makeRandomInitialEstimate(options.N, options.initSeed)));
      return 0;
    }

    const auto qcqpBuildStart = std::chrono::steady_clock::now();
    const QcqpProblem qcqpProblem(ringProblem.graph);
    const double qcqpBuildSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      qcqpBuildStart)
            .count();

    if (options.solverMode == SolverMode::Monolithic) {
      const auto sdpBuildStart = std::chrono::steady_clock::now();
      MosekMonolithicSDP sdp(qcqpProblem);
      const double sdpBuildSeconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        sdpBuildStart)
              .count();
      printBenchmarkResult(solveSdp(options, ringProblem.groundTruth, &sdp,
                                    qcqpBuildSeconds, sdpBuildSeconds));
      return 0;
    }

    const auto sdpBuildStart = std::chrono::steady_clock::now();
    MosekChordalSDP sdp(qcqpProblem, options.ordering);
    const double sdpBuildSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      sdpBuildStart)
            .count();
    BenchmarkResult result =
        solveSdp(options, ringProblem.groundTruth, &sdp, qcqpBuildSeconds,
                 sdpBuildSeconds);
    const auto [maxCliqueSize, numCliques] =
        chordalCliqueMetrics(sdp.bayesTree());
    result.maxCliqueSize = maxCliqueSize;
    result.numCliques = numCliques;
    printBenchmarkResult(result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
