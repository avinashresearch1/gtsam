/* Split RingSLAM, N=1000 by default. See Pose3RingSLAMComparison.md. */
#include "Pose3RingSLAMComparison.h"

int main(int argc, char** argv) {
  return ring_slam_example::run(argc, argv, "chordal");
}
