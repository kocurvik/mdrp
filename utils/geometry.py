import h5py
import numpy as np

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])

def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1)))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def get_pose(img1, img2, R_dict, T_dict):
    R1 = np.array(R_dict[img1])
    R2 = np.array(R_dict[img2])
    t1 = np.array(T_dict[img1])
    t2 = np.array(T_dict[img2])

    R = R2 @ R1.T
    t = -R @ t1 + t2
    return R, t


def get_gt_F(img1, img2, R_dict, T_dict, camera_dicts):
    R1, t1 = np.array(R_dict[img1]), np.array(T_dict[img1])
    R2, t2 = np.array(R_dict[img2]), np.array(T_dict[img2])
    R = R2 @ R1.T
    t = (t2 - R @ t1).ravel()
    cam1 = camera_dicts[img1]
    cam2 = camera_dicts[img2]
    K1 = np.array([[cam1['params'][0], 0.0, cam1['params'][-2]], [0.0, cam1['params'][0], cam1['params'][-1]], [0, 0, 1.0]])
    K2 = np.array([[cam2['params'][0], 0.0, cam2['params'][-2]], [0.0, cam2['params'][0], cam2['params'][-1]], [0, 0, 1.0]])

    return np.linalg.inv(K2.T) @ skew(t) @ R @ np.linalg.inv(K1)


def get_gt_E(img1, img2, R_dict, T_dict, camera_dicts):
    R1, t1 = np.array(R_dict[img1]), np.array(T_dict[img1])
    R2, t2 = np.array(R_dict[img2]), np.array(T_dict[img2])
    R = R2 @ R1.T
    t = (t2 - R @ t1).ravel()

    return skew(t) @ R


def get_inliers(F, x1, x2, threshold = 2.0):
    pts1 = np.column_stack([x1, np.ones(len(x1))])
    pts2 = np.column_stack([x2, np.ones(len(x2))])
    F_t = F.T
    line1_in_2 = pts1 @ F_t
    line2_in_1 = pts2 @ F

    # numerator = (x'^T F x) ** 2
    numerator = np.sum(pts2 * line1_in_2, axis=1) ** 2

    # denominator = (((Fx)_1**2) + (Fx)_2**2)) +  (((F^Tx')_1**2) + (F^Tx')_2**2))
    denominator = line1_in_2[:, 0] ** 2 + line1_in_2[:, 1] ** 2 + line2_in_1[:, 0] ** 2 + line2_in_1[:, 1] ** 2
    out = numerator / denominator
    return out < threshold ** 2


def add_rand_pts(x, cam_dict, multiplier):
    x_new = np.random.rand(int(multiplier * len(x)), 2)
    x_new[:, 0] *= cam_dict['width']
    x_new[:, 1] *= cam_dict['height']
    return np.row_stack([x, x_new])


def force_inliers(x1, x2, R, t, K1, K2, ratio, threshold):
    F = np.linalg.inv(K2.T) @ skew(t.ravel()) @ R @ np.linalg.inv(K1)
    inliers = get_inliers(F, x1, x2, threshold)
    x1, x2 = x1[inliers], x2[inliers]

    multiplier = (1 - ratio) / ratio

    x1_new = np.random.rand(int(multiplier * len(x1)), 2)
    x1_new[:, 0] *= K1[0, 2] / 2
    x1_new[:, 1] *= K1[1, 2] / 2

    x2_new = np.random.rand(int(multiplier * len(x2)), 2)
    x2_new[:, 0] *= K2[0, 2] / 2
    x2_new[:, 1] *= K2[1, 2] / 2

    return np.row_stack([x1, x1_new]), np.row_stack([x2, x2_new])


def get_camera_dicts(K_file_path):
    K_file = h5py.File(K_file_path)

    d = {}

    # Treat data from Charalambos differently since it is in pairs
    if 'K1_K2' in K_file_path:

        for k, v in K_file.items():
            key1, key2 = k.split('-')
            if key1 not in d.keys():
                K1 = np.array(v)[0, 0]
                d[key1] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K1[0, 2]), 'height': int(2 * K1[1,2]), 'params': [K1[0, 0], K1[0, 2], K1[1, 2]]}
            if key2 not in d.keys():
                K2 = np.array(v)[0, 1]
                d[key2] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K2[0, 2]), 'height': int(2 * K2[1,2]), 'params': [K2[0, 0], K2[0, 2], K2[1, 2]]}

        return d

    for key, v in K_file.items():
        K = np.array(v)
        d[key.replace('\\', '/')] = {'model': 'PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

    return d
