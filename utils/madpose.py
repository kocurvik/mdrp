import madpose

def madpose_opt_from_dict(d):
    options = madpose.HybridLORansacOptions()
    options.min_num_iterations = d['min_iterations']
    options.max_num_iterations = d['max_iterations']
    options.success_probability = 0.9999
    options.random_seed = 0  # for reproducibility
    options.final_least_squares = True
    options.threshold_multiplier = 5.0
    options.num_lo_steps = 4
    # squared px thresholds for reprojection error and epipolar error
    options.squared_inlier_thresholds = [d['max_reproj_error'] ** 2, d['max_epipolar_error'] ** 2]
    # weight when scoring for the two types of errors
    options.data_type_weights = [1.0, 1.0]

    est_config = madpose.EstimatorConfig()
    est_config.min_depth_constraint = True
    est_config.use_shift = True
    est_config.ceres_num_threads = 1

    return options, est_config
