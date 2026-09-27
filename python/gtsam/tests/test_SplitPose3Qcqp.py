"""The same split SE(3) graph supports vector and matrix QCQP conversions."""
import unittest

import numpy as np
import gtsam
from gtsam.symbol_shorthand import R, T


class TestSplitPose3Qcqp(unittest.TestCase):
    def test_point_round_trip_and_filter(self):
        point = np.array([1.2, -0.5, 2.3])
        lifted = gtsam.qcqpValuePoint3(point)
        np.testing.assert_allclose(lifted.reshape(-1), np.r_[1.0, point])
        np.testing.assert_allclose(gtsam.fromQcqpValuePoint3(-2 * lifted), point)
        values = gtsam.Values()
        gtsam.insertQcqpValuePoint3(T(0), point, values)
        gtsam.insertQcqpValueRot3(R(0), gtsam.Rot3(), values)
        points = gtsam.extractQcqpValuesPoint3(values)
        self.assertEqual(points.size(), 1)
        np.testing.assert_allclose(points.atPoint3(T(0)), point)
        with self.assertRaises(ValueError):
            gtsam.fromQcqpValuePoint3(np.zeros((4, 1)))

    def test_graph_cost_and_gauge(self):
        graph = gtsam.NonlinearFactorGraph()
        measured = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.2, -0.3, 0.4),
                               np.array([0.5, -0.2, 0.3]))
        graph.add(gtsam.FrobeniusBetweenFactorRot3(
            R(0), R(1), measured.rotation(),
            gtsam.noiseModel.Isotropic.Variance(3, 0.2)))
        graph.add(gtsam.RelativeTranslationFactor3(
            R(0), T(0), T(1), measured.translation(), 2.3))
        poses = [gtsam.Pose3(),
                 gtsam.Pose3(gtsam.Rot3.RzRyRx(-0.1, 0.2, 0.6),
                             np.array([1.3, -0.1, 0.8]))]
        typed, vector, matrix = gtsam.Values(), gtsam.Values(), gtsam.Values()
        for i, pose in enumerate(poses):
            typed.insert(R(i), pose.rotation())
            typed.insert(T(i), pose.translation())
            gtsam.insertQcqpValueRot3(R(i), pose.rotation(), vector)
            gtsam.insertQcqpValuePoint3(T(i), pose.translation(), vector)
            matrix.insert(R(i), pose.rotation().matrix().T)
            matrix.insert(T(i), pose.translation().reshape(1, 3))
        problem = gtsam.QcqpProblem(graph)
        self.assertAlmostEqual(problem.evaluate(vector)[0], graph.error(typed), 10)
        self.assertAlmostEqual(gtsam.QcqpProblem(graph, 3).evaluate(matrix)[0],
                               graph.error(typed), 10)
        problem.fixValue(R(0), gtsam.qcqpValueRot3(gtsam.Rot3()))
        problem.fixValue(T(0), gtsam.qcqpValuePoint3(np.zeros(3)))
        self.assertLess(problem.evaluate(vector)[1], 1e-12)
        self.assertEqual(graph.size(), 2)
        with self.assertRaises(ValueError):
            problem.fixValue(T(9), gtsam.qcqpValuePoint3(np.zeros(3)))

        transform = gtsam.Pose3(gtsam.Rot3.RzRyRx(0.7, -0.2, 0.5),
                                np.array([2.0, 3.0, -1.0]))
        shifted = gtsam.Values()
        for i, pose in enumerate(poses):
            pose = transform.compose(pose)
            shifted.insert(R(i), pose.rotation())
            shifted.insert(T(i), pose.translation())
        self.assertAlmostEqual(graph.error(typed), graph.error(shifted), 10)


if __name__ == '__main__':
    unittest.main()
