# Reproduce the split SE(3) ring with one C++ executable

Three examples share `Pose3RingSLAMComparison.h`, so the graph, measurements,
weights, initialization rule, and diagnostics have one implementation:

- `Pose3RingSLAM_Chordal.cpp`
- `Pose3RingSLAM_Monolithic.cpp`
- `Pose3RingSLAM_Staircase.cpp`

Each defaults to N=1000, seed 42, original noise, and 1000 seconds of native MOSEK
optimization. The staircase keeps its original rank cap of 8 and uses the GTSAM
implementation, not the upstream SE-Sync library. Chordal selects COLAMD.

With this GTSAM checkout configured as a Release build with MOSEK enabled:

```bash
cmake --build /external/gtsam-build --target \
  Pose3RingSLAM_Chordal Pose3RingSLAM_Monolithic Pose3RingSLAM_Staircase -j2
export MOSEKLM_LICENSE_FILE=/external/licenses/mosek.lic
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
mkdir -p /external/ring-results
/external/gtsam-build/examples/Pose3RingSLAM_Chordal \
  --N 1000 --max-time 1000 --output /external/ring-results/chordal1000
```

Replace `Chordal` with `Monolithic` or `Staircase` for the other executable.
The output prefix must be in an existing directory and should be unique. Output
includes measurement, pose, eigenvalue-ratio, and summary CSVs. Redirect stdout
and stderr to a log if running directly. The paper driver also captures native
MOSEK iterations, process timing, memory peaks and source/library hashes.

Linux `/proc` guards sample every 0.2 seconds: stop above 20 GiB RSS, below 4 GiB
system-available memory, or after the method's process limit (3600 seconds
monolithic, 1800 chordal, 1100 staircase). Sampled limits can overshoot. The paper
driver adds a 1000-second staircase optimizer-call guard. Non-Linux runs stop
before construction because these resource guards are Linux-specific.

Exit 0 means the example's checks pass; 2 means a returned iterate fails checks;
3 means a sampled resource guard fired; 1 means an exception. The original-noise
N=1000 chordal case is expected to return **2**, with Unknown/stall, recovered
objective about 87.4180622182 and SDP primal objective about 1.25405586024. The
paper's rank-one rule gives 5/1000 poses. This reproduces the recorded numerical
failure while fixing the memory blow-up. Monolithic can reach the memory guard;
staircase can finish without a rounded solution.

No prior is added to the relative-only nonlinear graph. Direct QCQP compilation
selects R(0)=I and t(0)=0. Both algorithms use the same objective; staircase uses
a different matrix lift and aligns recovered poses to the reference gauge.

See the [paper PR](https://github.com/avinashresearch1/Chordal_IJRR/pull/2) for
complete build instructions, the single reproduction script, the exact tested
source snapshot, native logs, all 33 final attempts, and the explanation of
clique-tree overlaps, shared homogenization, and fixed-value substitution.
