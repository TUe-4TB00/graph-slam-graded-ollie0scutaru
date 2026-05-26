
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # TODO: Add the odometry factor between X(4) and X(5) to the graph (BetweenFactorPose2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), gtsam.Pose2(np.sqrt(2), np.sqrt(2), np.pi/2), ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(5) and add it to the graph
    
    initial_estimate.insert(X(4), gtsam.Pose2(4 + np.sqrt(2), np.sqrt(2), np.pi/2))

    return graph, initial_estimate