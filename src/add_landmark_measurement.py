import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    pose = result.atPose2(X(4))
    landmark = result.atPoint2(L(2))

    dx = landmark[0] - pose.x()
    dy = landmark[1] - pose.y()

    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    rotation = np.arctan2(dy, dx)- pose.theta()
    distance = np.sqrt(dx**2 + dy**2)
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromAngle(rotation), distance, MEASUREMENT_NOISE))
    return graph