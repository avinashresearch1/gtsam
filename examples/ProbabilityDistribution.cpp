#include <gtsam/discrete/DiscreteDistribution.h>


// In GTSAM, measurement functions are represented as 'factors'. Several common factors
// have been provided with the library for solving robotics/SLAM/Bundle Adjustment problems.
// Here we will use Between factors for the relative motion described by odometry measurements.
// Also, we will initialize the robot at the origin using a Prior factor.
#include <gtsam/slam/BetweenFactor.h>

// When the factors are created, we will add them to a Factor Graph. As the factors we are using
// are nonlinear factors, we will need a Nonlinear Factor Graph.
#include <gtsam/nonlinear/NonlinearFactorGraph.h>


using namespace std;
using namespace gtsam;

int main(int argc, char** argv) {
  // create discrete probability distribution: 
  DiscreteDistribution distribution = DiscreteDistribution(DiscreteKey(42, 3), "0.4/0.1/0.5");

  distribution.print("Discrete Distribution: ");

  return 0;
}
