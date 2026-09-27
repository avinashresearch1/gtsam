/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010-2026, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * Compare direct SDPs and the GTSAM synchronization staircase on one immutable
 * split SE(3) ring graph. This uses the staircase from the certifiable PGO
 * notebooks, not the upstream SE-Sync library. No separate LM baseline is run.
 * Shared implementation for the three split RingSLAM example executables.
 * Defaults: N=1000, original noise, 1000-second native MOSEK time limit.
 * Use the paper runner for RSS and process-time guards.
 */
#include <gtsam/certifiable/LiftedSDPProblem.h>
#include <gtsam/certifiable/RiemannianStaircaseOptimizer.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/FrobeniusFactor.h>
#include <gtsam/slam/RelativeTranslationFactor.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#pragma once

namespace ring_slam_example {

using namespace gtsam;
using symbol_shorthand::R;
using symbol_shorthand::T;
using Clock = std::chrono::steady_clock;

/** Read one Linux memory counter, expressed in KiB. */
inline size_t memoryCounter(const char* path, const std::string& field) {
  std::ifstream stream(path);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind(field, 0) == 0)
      return std::stoull(line.substr(field.size()));
  }
  throw std::runtime_error("Cannot read Linux memory counter: " + field);
}

/** Sample memory and wall time so a standalone large example can stop safely. */
class ResourceGuard {
  std::atomic<bool> stopped_{false};
  std::thread thread_;
 public:
  /** Start the same 20 GiB RSS and 4 GiB available-memory guards as the paper. */
  explicit ResourceGuard(double wallLimit) {
#ifdef __linux__
    memoryCounter("/proc/self/status", "VmRSS:");
    memoryCounter("/proc/meminfo", "MemAvailable:");
    thread_ = std::thread([this, wallLimit]() {
      const auto start = Clock::now();
      while (!stopped_) {
        try {
          const auto rss = memoryCounter("/proc/self/status", "VmRSS:");
          const auto available = memoryCounter("/proc/meminfo", "MemAvailable:");
          const char* reason = nullptr;
          if (available < 4ULL * 1024 * 1024) reason = "system_memory_floor";
          else if (rss > 20ULL * 1024 * 1024) reason = "rss_limit";
          else if (std::chrono::duration<double>(Clock::now()-start).count() > wallLimit)
            reason = "process_timeout";
          if (reason) {
            std::cerr << "RESOURCE_LIMIT " << reason << " rss_kib=" << rss << std::endl;
            std::_Exit(3);
          }
        } catch (const std::exception& error) {
          std::cerr << "RESOURCE_LIMIT monitor_failure: " << error.what() << std::endl;
          std::_Exit(3);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    });
#else
    throw std::runtime_error("These guarded paper examples currently require Linux.");
#endif
  }
  /** Stop monitoring after the example finishes or throws. */
  ~ResourceGuard() {
    stopped_ = true;
    if (thread_.joinable()) thread_.join();
  }
};

// All algorithms consume the same graph object and stored measurements.
struct Ring {
  NonlinearFactorGraph graph;
  std::vector<Pose3> truth;
  std::vector<Pose3> odometry;
};

/** Build the fixed-seed split SE(3) ring and its immutable measurements. */
inline Ring makeRing(size_t count, double noiseScale) {
  Ring ring;
  constexpr double pi = 3.14159265358979323846;
  constexpr double kappa = 100.0, tau = 25.0;
  std::mt19937 generator(42);
  std::normal_distribution<double> normal;
  for (size_t i = 0; i < count; ++i) {
    const double theta = 2.0 * pi * i / count;
    ring.truth.emplace_back(
        Rot3::RzRyRx(0.15 * std::sin(theta), 0.12 * std::cos(2 * theta),
                     theta + pi / 2),
        Point3(5 * std::cos(theta), 5 * std::sin(theta),
               0.5 * std::sin(2 * theta)));
  }
  const Pose3 first = ring.truth.front();
  for (auto& pose : ring.truth) pose = first.between(pose);
  ring.odometry.emplace_back();
  for (size_t i = 0; i < count; ++i) {
    const size_t j = (i + 1) % count;
    const Pose3 relative = ring.truth[i].between(ring.truth[j]);
    Vector3 omega, translationNoise;
    for (int k = 0; k < 3; ++k) {
      omega(k) = noiseScale * normal(generator) / std::sqrt(2 * kappa);
      translationNoise(k) = noiseScale * normal(generator) / std::sqrt(tau);
    }
    const Pose3 measured(relative.rotation().compose(Rot3::Expmap(omega)),
                         relative.translation() + translationNoise);
    ring.graph.emplace_shared<FrobeniusBetweenFactor<Rot3>>(
        R(i), R(j), measured.rotation(),
        noiseModel::Isotropic::Variance(3, 1 / kappa));
    ring.graph.emplace_shared<RelativeTranslationFactor3>(
        R(i), T(i), T(j), measured.translation(), tau);
    if (j != 0) ring.odometry.push_back(ring.odometry.back().compose(measured));
  }
  return ring;
}

/** Represent poses as separate Rot3 and Point3 values. */
inline Values splitValues(const std::vector<Pose3>& poses) {
  Values values;
  for (size_t i = 0; i < poses.size(); ++i) {
    values.insert(R(i), poses[i].rotation());
    values.insert(T(i), poses[i].translation());
  }
  return values;
}

// Measure complete rotations and translations in the first-pose gauge.
/** Align all recovered poses to the first pose. */
inline void normalize(std::vector<Pose3>* poses) {
  const Pose3 first = poses->front();
  for (auto& pose : *poses) pose = first.between(pose);
}

struct Summary {
  std::string method;
  double objective = 0, relaxation = 0, build = 0, solve = 0, recovery = 0;
  double optimizer = 0;
  double violation = 0, stationarity = 0, minEigenvalue = 0;
  bool certified = false;
  size_t rank = 0;
  std::string status;
  std::vector<double> evrs;
  KeyVector evrKeys;
  std::vector<Pose3> poses;
};

/** Return wall seconds since the supplied time point. */
inline double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

// The reference constraints are a coordinate choice in the compiled problem.
/** Compile the graph and select the first-pose gauge for direct SDP. */
inline QcqpProblem compile(const NonlinearFactorGraph& graph) {
  QcqpProblem problem(graph);
  problem.fixValue(R(0), qcqpValue(Rot3()));
  problem.fixValue(T(0), qcqpValue(Point3(0, 0, 0)));
  return problem;
}

/** Optimize a direct SDP and recover graph poses and diagnostics. */
template <typename Solver>
void solveDirect(const Ring& ring, Solver* solver, Summary* summary,
                 double maxTime) {
  const auto start = Clock::now();
  std::map<std::string, double> options{{"intpntCoTolRelGap", 1e-10},
                                        {"intpntCoTolPfeas", 1e-10},
                                        {"intpntCoTolDfeas", 1e-10}};
  if (maxTime > 0) options["optimizerMaxTime"] = maxTime;
  solver->solve(options);
  summary->solve = elapsed(start);
  summary->optimizer = solver->solveTimeSeconds();
  const auto recoveryStart = Clock::now();
  summary->status = solver->problemStatus();
  summary->relaxation = solver->objectiveValue();
  summary->evrs = solver->variableEVRs();
  summary->evrKeys = solver->orderedKeys();
  const Values recovered = solver->qcqpValues();
  const Values rotations = extractQcqpValues<Rot3>(recovered);
  const Values points = extractQcqpValues<Point3>(recovered);
  for (size_t i = 0; i < ring.truth.size(); ++i) {
    summary->poses.emplace_back(rotations.at<Rot3>(R(i)),
                                points.at<Point3>(T(i)));
  }
  // Check the raw homogeneous recovery before projection and gauge alignment.
  summary->violation =
      compile(ring.graph).eConstraints().violationNorm(recovered);
  normalize(&summary->poses);
  summary->objective = ring.graph.error(splitValues(summary->poses));
  summary->recovery = elapsed(recoveryStart);
}

/** Run the original staircase settings and recover its rounded poses. */
inline Summary solveStaircase(const Ring& ring) {
  Summary summary;
  summary.method = "gtsam_staircase";
  const auto start = Clock::now();
  Values initial;
  for (size_t i = 0; i < ring.odometry.size(); ++i) {
    initial.insert(R(i),
                   Matrix(ring.odometry[i].rotation().matrix().transpose()));
    initial.insert(T(i), Matrix(ring.odometry[i].translation().transpose()));
  }
  RiemannianStaircaseParams params;
  params.verbose = true;
  params.pMin = 3;
  params.pMax = 8;
  params.eta = 1e-7;
  params.spectraTol = 1e-9;
  params.almParams->maxIterations = 100;
  params.almParams->absoluteViolationTolerance = 1e-9;
  params.almParams->absoluteStationarityTolerance = 1e-7;
  params.almParams->bclInitialPenalty = 1000.0;
  params.almParams->bclBetaOmega = 0.1;
  params.almParams->lmParams.maxIterations = 500;
  params.almParams->lmParams.absoluteErrorTol = 1e-12;
  params.almParams->lmParams.relativeErrorTol = 1e-12;
  RiemannianStaircaseOptimizer solver(ring.graph, initial, params);
  summary.build = elapsed(start);
  const auto solveStart = Clock::now();
  std::cout << "PHASE staircase_optimizer_start" << std::endl;
  const auto result = solver.optimize();
  summary.solve = elapsed(solveStart);
  summary.optimizer = summary.solve;
  const auto recoveryStart = Clock::now();
  summary.certified = result.certified;
  summary.rank = result.finalRank;
  summary.minEigenvalue = result.minEigenvalue;
  summary.stationarity = result.stationarityPerLevel.back();
  summary.relaxation = result.costPerLevel.back();
  summary.violation = QcqpProblem(ring.graph, result.finalRank)
                          .eConstraints()
                          .violationNorm(result.values);
  summary.status = result.certified ? "certified_sdp" : "uncertified";
  std::cout << "STAIRCASE_RESULT certified=" << result.certified << " final_rank=" << result.finalRank << " rounded=" << result.hasRoundedSolution() << " violation=" << summary.violation << " optimizer_s=" << summary.solve << std::endl;
  for(size_t k=0;k<result.ranksVisited.size();++k) std::cout << "LEVEL rank=" << result.ranksVisited[k] << " cost=" << result.costPerLevel[k] << " stationarity=" << result.stationarityPerLevel[k] << " min_eigenvalue=" << result.minEigenvaluePerLevel[k] << " nlp_s=" << result.nlpTimePerLevel[k] << " verify_s=" << result.verifyTimePerLevel[k] << std::endl;
  if (!result.hasRoundedSolution()) {
    throw std::runtime_error("Staircase has no rounded solution.");
  }
  const Values rounded = result.roundedValues();
  size_t reflections = 0;
  for (size_t i = 0; i < ring.truth.size(); ++i) {
    if (rounded.at<Matrix>(R(i)).determinant() < 0) ++reflections;
  }
  Matrix3 reflection = Matrix3::Identity();
  if (reflections > ring.truth.size() / 2) reflection(2, 2) = -1;
  for (size_t i = 0; i < ring.truth.size(); ++i) {
    const Matrix3 rotation =
        (rounded.at<Matrix>(R(i)) * reflection).transpose();
    const Point3 point = (rounded.at<Matrix>(T(i)) * reflection).transpose();
    summary.poses.emplace_back(Rot3::ClosestTo(rotation), point);
  }
  normalize(&summary.poses);
  summary.objective = ring.graph.error(splitValues(summary.poses));
  summary.recovery = elapsed(recoveryStart);
  return summary;
}

/** Write authoritative full-precision measurements. */
inline void writeMeasurements(const Ring& ring, const std::string& output) {
  const size_t count = ring.truth.size();
  std::ofstream measurements(output + "_measurements.csv");
  if (!measurements) throw std::runtime_error("Cannot open measurements file");
  measurements << std::setprecision(17)
               << "source,target,kappa,tau,tx,ty,tz,r00,r01,r02,r10,r11,r12,"
                  "r20,r21,r22\n";
  for (size_t i = 0; i < count; ++i) {
    const auto rotation =
        std::dynamic_pointer_cast<FrobeniusBetweenFactor<Rot3>>(
            ring.graph.at(2 * i));
    const auto translation =
        std::dynamic_pointer_cast<RelativeTranslationFactor3>(
            ring.graph.at(2 * i + 1));
    measurements << i << ',' << (i + 1) % count << ",100,25";
    for (double coordinate : translation->measured())
      measurements << ',' << coordinate;
    const Matrix3 matrix = rotation->measured().matrix();
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) measurements << ',' << matrix(row, col);
    measurements << '\n';
  }
}

/** Check D=1 and D=3 compiled costs against the nonlinear graph. */
inline void verifyCompilations(const Ring& ring) {
  const size_t count = ring.truth.size();
  // Verify both compilations at a nonoptimal assignment before benchmarking.
  Values vectorValues, matrixValues;
  for (size_t i = 0; i < count; ++i) {
    insertQcqpValue(R(i), ring.odometry[i].rotation(), vectorValues);
    insertQcqpValue(T(i), ring.odometry[i].translation(), vectorValues);
    matrixValues.insert(
        R(i), Matrix(ring.odometry[i].rotation().matrix().transpose()));
    matrixValues.insert(T(i),
                        Matrix(ring.odometry[i].translation().transpose()));
  }
  const double initialCost = ring.graph.error(splitValues(ring.odometry));
  for (const auto& item : std::vector<std::pair<int, Values>>{
           {1, vectorValues}, {3, matrixValues}}) {
    const double compiled =
        QcqpProblem(ring.graph, item.first).costs().error(item.second);
    if (std::abs(initialCost - compiled) > 1e-8 * std::max(1.0, initialCost)) {
      throw std::runtime_error("Graph/QCQP cost mismatch");
    }
  }
}

/** Write solver diagnostics and test recovery accuracy. */
inline bool reportSummary(const Summary& summary, const Summary& reference,
                   double noiseScale, std::ostream& csv, std::ostream& poses,
                   std::ostream& evrs, bool compare) {
  const size_t count = summary.poses.size();
  const std::string& method = summary.method;
  double rotationDifference = 0, translationDifference = 0;
  for (size_t i = 0; i < count; ++i) {
    rotationDifference = std::max(
        rotationDifference, Rot3::Logmap(reference.poses[i].rotation().between(
                                             summary.poses[i].rotation()))
                                .norm());
    translationDifference = std::max(
        translationDifference,
        (reference.poses[i].translation() - summary.poses[i].translation())
            .norm());
    poses << method << ',' << i;
    for (double coordinate : summary.poses[i].translation())
      poses << ',' << coordinate;
    const Matrix3 rotation = summary.poses[i].rotation().matrix();
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) poses << ',' << rotation(row, col);
    poses << '\n';
  }
  double minEvr = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < summary.evrs.size(); ++i) {
    minEvr = std::min(minEvr, summary.evrs[i]);
    evrs << method << ',' << DefaultKeyFormatter(summary.evrKeys[i]) << ','
         << summary.evrs[i] << '\n';
  }
  const double total = summary.build + summary.solve + summary.recovery;
  csv << count << ',' << noiseScale << ',' << method << ',' << summary.objective
      << ',' << summary.relaxation << ',' << summary.build << ','
      << summary.solve << ',' << summary.recovery << ',' << total << ',';
  if (!summary.evrs.empty()) csv << minEvr;
  csv << ',';
  if (compare) csv << rotationDifference;
  csv << ',';
  if (compare) csv << translationDifference;
  csv << ',' << summary.violation << ',' << summary.certified << ',' << summary.rank
      << ',' << summary.stationarity << ',' << summary.minEigenvalue << ','
      << summary.status << ',' << summary.optimizer << std::endl;
  std::cout << std::setprecision(12) << method
            << " objective=" << summary.objective
            << " relaxation=" << summary.relaxation << " minEVR=" << minEvr
            << " rotationDiff=" << rotationDifference
            << " translationDiff=" << translationDifference
            << " total=" << total << std::endl;
  const double tolerance = 1e-5 * std::max(1.0, std::abs(reference.objective));
  return std::isfinite(summary.objective) &&
         std::abs(summary.objective - reference.objective) <= tolerance &&
         std::abs(summary.objective - summary.relaxation) <= tolerance &&
         rotationDifference <= 1e-3 && translationDifference <= 1e-3 &&
         std::isfinite(summary.violation) && summary.violation <= 1e-5 &&
         (summary.evrs.empty()
              ? summary.certified
              : std::all_of(summary.evrs.begin(), summary.evrs.end(),
                            [](double ratio) {
                              return std::isfinite(ratio) && ratio >= 1e5;
                            }));
}

/** Run the selected algorithm; return 2 for failed checks and 3 for a resource stop. */
inline int run(int argc, char** argv, const std::string& selectedMethod) {
  try {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
      std::cout << "Usage: " << argv[0]
                << " [--N 1000] [--noise-scale 1] [--max-time 1000] [--output prefix]\n"
                << "Method: " << selectedMethod << "; Linux memory guards enabled.\n";
      return 0;
    }
    if (argc % 2 == 0) throw std::invalid_argument("Every option requires a value; use --help.");
    size_t count = 1000;
    double noiseScale = 1.0;
    double maxTime = 1000.0;
    std::string output = "ring1000_" + selectedMethod;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 >= argc) throw std::invalid_argument("Missing option value");
      const std::string option = argv[i];
      if (option == "--N")
        count = std::stoul(argv[i + 1]);
      else if (option == "--noise-scale")
        noiseScale = std::stod(argv[i + 1]);
      else if (option == "--max-time")
        maxTime = std::stod(argv[i + 1]);
      else if (option == "--output")
        output = argv[i + 1];
      else
        throw std::invalid_argument("Unknown option " + option);
    }
    if (count < 3 || !std::isfinite(noiseScale) || noiseScale < 0) {
      throw std::invalid_argument("Require N >= 3 and finite noise-scale >= 0");
    }
    if (!std::isfinite(maxTime) || maxTime < 0 ||
        (selectedMethod != "all" && selectedMethod != "monolithic" &&
         selectedMethod != "chordal" && selectedMethod != "gtsam_staircase")) {
      throw std::invalid_argument("Invalid method or max-time");
    }
    ResourceGuard resources(selectedMethod == "monolithic" ? 3600.0 :
                            selectedMethod == "chordal" ? 1800.0 : 1100.0);
    const Ring ring = makeRing(count, noiseScale);
    writeMeasurements(ring, output);
    verifyCompilations(ring);
    std::ofstream csv(output + ".csv"), poses(output + "_poses.csv"),
        evrs(output + "_evrs.csv");
    if (!csv || !poses || !evrs)
      throw std::runtime_error("Cannot open output files");
    csv << std::setprecision(17);
    poses << std::setprecision(17);
    evrs << std::setprecision(17);
    csv << "N,noise_scale,method,objective,relaxation_objective,build_s,solve_"
           "s,recovery_s,total_s,min_evr,max_rotation_difference_rad,max_"
           "translation_difference,raw_constraint_violation,certified,rank,"
           "stationarity,min_eigenvalue,status,optimizer_s\n";
    poses << "method,index,tx,ty,tz,r00,r01,r02,r10,r11,r12,r20,r21,r22\n";
    evrs << "method,key,evr\n";
    Summary reference;
    bool passed = true;
    for (const std::string method :
         {"monolithic", "chordal", "gtsam_staircase"}) {
      if (selectedMethod != "all" && selectedMethod != method) continue;
      std::cout << "N=" << count << " starting " << method << std::endl;
      Summary summary;
      summary.method = method;
      if (method == "gtsam_staircase")
        summary = solveStaircase(ring);
      else {
        const auto start = Clock::now();
        const QcqpProblem problem = compile(ring.graph);
        std::cout << "PHASE qcqp_ready construction_start elapsed_s=" << elapsed(start) << std::endl;
        if (method == "monolithic") {
          MosekMonolithicSDP solver(problem);
          summary.build = elapsed(start);
          std::cout << "PHASE construction_finished optimizer_start elapsed_s=" << summary.build << std::endl;
          solveDirect(ring, &solver, &summary, maxTime);
        } else {
          MosekChordalSDP solver(problem, ChordalOrderingType::Colamd);
          summary.build = elapsed(start);
          std::cout << "PHASE construction_finished optimizer_start elapsed_s=" << summary.build << std::endl;
          solveDirect(ring, &solver, &summary, maxTime);
        }
      }
      if (reference.poses.empty()) reference = summary;
      passed =
          reportSummary(summary, reference, noiseScale, csv, poses, evrs,
                        selectedMethod == "all") &&
          passed;
    }
    std::cout << (passed ? "PASS" : "FAIL") << " N=" << count << std::endl;
    return passed ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}

}  // namespace ring_slam_example
