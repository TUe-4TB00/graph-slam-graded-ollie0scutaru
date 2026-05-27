import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()

    # TODO: Initialize the optimizer 
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()

    return result


def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    
    ind_best_pose = ["a", "b", "c", "d"]
    sum_of_marginals = [0, 0]
    best_score = [float("inf"), float("inf")]
    best_pose_copy = [None, None]

    for j in [1,2]:
        landmark_sum = 0
        for i in range(len(pose_options)):
            graph_copy = gtsam.NonlinearFactorGraph(graph)
            estimate = gtsam.Values(initial_estimate)
            
            best_pose = ind_best_pose[i]      # chosen pose option
            best_landmark = j    # chosen landmark (1 or 2)
            pose_5 = pose_options[best_pose]
            graph_copy, estimate = add_pose(graph_copy, estimate, pose_5)
            result = optimize(graph_copy, estimate)
            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, best_landmark)
            result = optimize(graph_copy, estimate)
    
            # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(graph_copy, result)

            score = marginals.marginalCovariance(L(j)).sum()

            if score < best_score[j-1]:
                best_score[j-1] = score
                best_pose_copy[j-1] = ind_best_pose[i]  

            landmark_sum += score      

        sum_of_marginals[j-1] = landmark_sum
            
        # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    
    ind_best = np.argmin(sum_of_marginals)

    best_landmark = ind_best + 1
    best_pose = best_pose_copy[ind_best]

    return best_pose, best_landmark, sum_of_marginals[ind_best]


def minimize_errors(graph_copy_no_X4, initial_estimate, pose_options):
    
    ind_best_pose = ["a", "b", "c", "d"]
    min_error = float("inf")
    best_pose_copy = [None, None]
    best_pose = None
    best_landmark = None
    values_of_x = [0, 2, 4]

    for i in range(len(pose_options)):
        for j in [1,2]:
            graph_copy = gtsam.NonlinearFactorGraph(graph_copy_no_X4)
            estimate = gtsam.Values(initial_estimate)
            
            best_pose = ind_best_pose[i]      # chosen pose option
            best_landmark = j    # chosen landmark (1 or 2)
            pose_5 = pose_options[best_pose]
            graph_copy, estimate = add_pose(graph_copy, estimate, pose_5)
            result = optimize(graph_copy, estimate)
            graph_copy = add_landmark_measurement(graph_copy, result, pose_5, j)
            result = optimize(graph_copy, estimate)

            total_error = sum(abs(result.atPose2(X(i)).x() - true)
                + abs(result.atPose2(X(i)).y())
                + abs(result.atPose2(X(i)).theta())
                for i, true in zip([1, 2, 3], values_of_x)
                )

            if total_error < min_error:
                min_error = total_error
                best_pose = i
                best_landmark = j

    return best_pose, best_landmark, min_error