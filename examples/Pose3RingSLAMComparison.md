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
3 means a sampled resource guard fired; 1 means an exception. Unknown status is
never promoted to a certificate. Monolithic can still reach the memory guard.

The direct SDP algorithms now append five column-orthogonality equations and
six cyclic cross-product equations to the existing Rot3 QCQP constraints. These
are identities on SO(3): they do not change the nonlinear factor graph or its
feasible poses, but they strengthen its relaxation. One redundant column-norm
equation is omitted because the row constraints already fix the trace.

By default, **all three algorithms use the same local initializer**, obtained by
LM on the unchanged nonlinear graph from odometry. Use `--initialization odometry`
to skip that local solve. The initializer's time is included in construction
and whole-process timings, not MOSEK or staircase optimizer timings.

Direct methods use the initializer translations as coordinate origins,
`t_i = c_i + u_i`, compiling each cost as `S.transpose() * Q * S`. Fixed-gauge
origins are zero. They recover each original-coordinate moment matrix by the
inverse congruence before extracting values or checking rank. The public
`momentMatrix(key)` accessor supports this for both SDP formulations.
Rotations are projected during recovery; **no LM refinement follows the SDP**.
The program checks the transformed cost and constraints against the original
graph at a feasible nonoptimal assignment before solving.

The eigenvalue ratio uses the largest eigenvalue divided by the largest
absolute nonprincipal eigenvalue (with a machine-epsilon floor). This prevents
a negative second eigenvalue from concealing a non-PSD block. A pose passes
only when both its original-coordinate rotation and translation ratios exceed
1e5. The raw constraint and recovery-gap tolerances remain unchanged.

This addresses the original-noise N=1000 case that previously returned 5/1000
rank-one poses and objective 87.418. The centered, strengthened experiment
recovers objective approximately 1.30626306625 with 1000/1000 rank-one poses.
See the paper results for native statuses, numerical residuals and separate
comparisons against the original and locally initialized staircase.

No prior is added to the relative-only nonlinear graph. Direct QCQP compilation
selects R(0)=I and t(0)=0. Both algorithms use the same objective; staircase uses
a different matrix lift and aligns recovered poses to the reference gauge.

See the [paper PR](https://github.com/avinashresearch1/Chordal_IJRR/pull/2) for
complete build instructions, the single reproduction script, the exact tested
source snapshot, native logs, all 33 final attempts, and the explanation of
clique-tree overlaps, shared homogenization, and fixed-value substitution.
