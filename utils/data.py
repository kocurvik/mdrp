import numpy as np

basenames_eth = ['courtyard_splg', 'delivery_area_splg', 'electro_splg', 'facade_splg', 'kicker_splg', 'meadow_splg', 'office_splg', 'pipes_splg', 'playground_splg', 'relief_2_splg', 'relief_splg', 'terrace_splg', 'terrains_splg']
basenames_pt_old = ['british_museum_splg', 'florence_cathedral_splg', 'lincoln_memorial_splg', 'london_bridge_splg', 'milan_cathedral_splg', 'mount_rushmore_splg', 'sagrada_familia_splg', 'stpauls_cathedral_splg']
basenames_pt = ['florence_cathedral_splg', 'lincoln_memorial_splg', 'london_bridge_splg', 'milan_cathedral_splg', 'sagrada_familia_splg']
basenames_all = basenames_eth + basenames_pt


def get_basenames(dataset):
    if dataset == 'pt':
        return basenames_pt
    elif dataset == 'eth':
        return basenames_eth
    elif dataset == 'all':
        return basenames_all


def get_valid_depth_mask(d):
    l = np.logical_or(np.isinf(d[:, 0]), np.isinf(d[:, 1]))
    l = np.logical_or(np.isnan(d[:, 0]), l)
    l = np.logical_or(np.isnan(d[:, 1]), l)
    l = np.logical_or(d[:, 0] < 0, l)
    l = np.logical_or(d[:, 1] < 0, l)
    return l

def depth_indices(depth):
    if depth == 1: # real
        return (8, 9)
    elif depth == 2: # midas
        return (10, 11)
    elif depth == 3:  # dpt
        return (12, 13)
    elif depth == 4: # zoe
        return (14, 15)
    elif depth == 5: # depth anyV1B
        return (16, 17)
    elif depth == 6:  # depth anyV2B
        return (18, 19)
    elif depth == 7: # apple depth pro
        return (20, 21)
    elif depth == 8:  # metric 3d
        return (22, 23)
    elif depth == 9:  # marigold e2e
        return (24, 25)
    elif depth == 10:  # moge
        return (26, 27)
    elif depth == 11:  # marigold
        return (28, 29)
    elif depth == 12:  # unidepth
        return (30, 31)


def R_err_fun(r):
    R_gt = np.array(r['R_gt'])
    R = np.array(r['R'])
    # R2R1 = np.dot(R_gt, np.transpose(R))
    # cos_angle = max(min(1.0, 0.5 * (np.trace(R2R1) - 1.0)), -1.0)
    # err_r = np.rad2deg(np.acos(cos_angle))
    # R2R1 = R.T @ R_gt
    # err_r = np.rad2deg(np.arccos(np.clip((np.trace(R2R1) - 1) / 2, -1, 1)))

    sin_angle1 = np.linalg.norm(R_gt - R) / (2 * np.sqrt(2))
    sin_angle = max(min(1.0, sin_angle1), -1.0)
    err_r = np.rad2deg(2*np.arcsin(sin_angle))
    return err_r


def t_err_fun(r):
    t = np.array(r['t']).flatten()
    t_gt = np.array(r['t_gt']).flatten()

    eps = 1e-15
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt) ** 2))
    err_t = np.rad2deg(np.arccos(np.sqrt(1 - loss_t)))

    # err_t = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / (np.linalg.norm(t) * np.linalg.norm(t_gt)), -1, 1)))

    # t = t / (np.linalg.norm(t))
    # t_gt = t_gt / (np.linalg.norm(t_gt))
    # err_t = np.rad2deg(2*np.arcsin(np.linalg.norm(t - t_gt)*0.5))

    return err_t

def err_fun_pose(r):
    return max(R_err_fun(r), t_err_fun(r))
