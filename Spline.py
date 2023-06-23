import torch
import numpy as np

delt = 0


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack(
        [
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1),
        ],
        dim=-2,
    )
    return wx


def taylor_A(x, nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        if i > 0:
            denom *= (2 * i) * (2 * i + 1)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_B(x, nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 1) * (2 * i + 2)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def taylor_C(x, nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.0
    for i in range(nth + 1):
        denom *= (2 * i + 2) * (2 * i + 3)
        ans = ans + (-1) ** i * x ** (2 * i) / denom
    return ans


def exp_r2q_parallel(r, eps=1e-9):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    theta = 0.5 * torch.sqrt(x**2 + y**2 + z**2)
    bool_criterion = (theta < eps).unsqueeze(-1).repeat(1, 1, 4)
    return torch.where(
        bool_criterion, exp_r2q_taylor(x, y, z, theta), exp_r2q(x, y, z, theta)
    )


def exp_r2q(x, y, z, theta):
    lambda_ = torch.sin(theta) / (2.0 * theta)
    qx = lambda_ * x
    qy = lambda_ * y
    qz = lambda_ * z
    qw = torch.cos(theta)
    return torch.stack([qx, qy, qz, qw], -1)


def exp_r2q_taylor(x, y, z, theta):
    qx = (1.0 / 2.0 - 1.0 / 12.0 * theta**2 - 1.0 / 240.0 * theta**4) * x
    qy = (1.0 / 2.0 - 1.0 / 12.0 * theta**2 - 1.0 / 240.0 * theta**4) * y
    qz = (1.0 / 2.0 - 1.0 / 12.0 * theta**2 - 1.0 / 240.0 * theta**4) * z
    qw = 1.0 - 1.0 / 2.0 * theta**2 + 1.0 / 24.0 * theta**4
    return torch.stack([qx, qy, qz, qw], -1)


def q_to_R_parallel(q):
    qb, qc, qd, qa = q.unbind(dim=-1)
    R = torch.stack(
        [
            torch.stack(
                [
                    1 - 2 * (qc**2 + qd**2),
                    2 * (qb * qc - qa * qd),
                    2 * (qa * qc + qb * qd),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qb * qc + qa * qd),
                    1 - 2 * (qb**2 + qd**2),
                    2 * (qc * qd - qa * qb),
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    2 * (qb * qd - qa * qc),
                    2 * (qa * qb + qc * qd),
                    1 - 2 * (qb**2 + qc**2),
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    return R


def q_to_Q_parallel(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    Q_0 = torch.stack([w, -z, y, x], -1).unsqueeze(-2)
    Q_1 = torch.stack([z, w, -x, y], -1).unsqueeze(-2)
    Q_2 = torch.stack([-y, x, w, z], -1).unsqueeze(-2)
    Q_3 = torch.stack([-x, -y, -z, w], -1).unsqueeze(-2)
    Q_ = torch.cat([Q_0, Q_1, Q_2, Q_3], -2)
    return Q_


def q_to_q_conj_parallel(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    q_conj_ = torch.stack([-x, -y, -z, w], -1)
    return q_conj_


def log_q2r_parallel(q, eps_theta=1e-20, eps_w=1e-10):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    theta = torch.sqrt(x**2 + y**2 + z**2)

    bool_theta_0 = theta < eps_theta
    bool_w_0 = torch.abs(w) < eps_w
    bool_w_0_left = torch.logical_and(bool_w_0, w < 0)

    lambda_ = torch.where(
        bool_w_0,
        torch.where(
            bool_w_0_left,
            log_q2r_lim_w_0_left(theta),
            log_q2r_lim_w_0_right(theta)
        ),
        torch.where(
            bool_theta_0,
            log_q2r_taylor_theta_0(w, theta),
            log_q2r(w, theta)
        ),
    )

    r_ = torch.stack([lambda_ * x, lambda_ * y, lambda_ * z], -1)

    return r_


def log_q2r(w, theta):
    return 2.0 * (torch.arctan(theta / w)) / theta


def log_q2r_taylor_theta_0(w, theta):
    return 2.0 / w - 2.0 / 3.0 * (theta**2) / (w * w * w)


def log_q2r_lim_w_0_left(theta):
    return -torch.pi / theta


def log_q2r_lim_w_0_right(theta):
    return torch.pi / theta


def SE3_to_se3(Rt, eps=1e-8):  # [...,3,4]
    R, t = Rt.split([3, 1], dim=-1)
    w = SO3_to_so3(R)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    invV = I - 0.5 * wx + (1 - A / (2 * B)) / (theta**2 + eps) * wx @ wx
    u = (invV @ t)[..., 0]
    wu = torch.cat([w, u], dim=-1)
    return wu


def SO3_to_so3(R, eps=1e-7):  # [...,3,3]
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()[
        ..., None, None
    ] % np.pi  # ln(R) will explode if theta==pi
    lnR = (
        1 / (2 * taylor_A(theta) + 1e-8) * (R - R.transpose(-2, -1))
    )  # FIXME: wei-chiu finds it weird
    w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
    w = torch.stack([w0, w1, w2], dim=-1)
    return w


def se3_to_SE3(wu):  # [...,3]
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)  # wx=[0 -w(2) w(1);w(2) 0 -w(0);-w(1) w(0) 0]
    theta = w.norm(dim=-1)[..., None, None]  # theta=sqrt(w'*w)
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    Rt = torch.cat([R, (V @ u[..., None])], dim=-1)
    return Rt


def SE3_to_se3_N(poses_rt):
    poses_se3_list = []
    for i in range(poses_rt.shape[0]):
        pose_se3 = SE3_to_se3(poses_rt[i])
        poses_se3_list.append(pose_se3)
    poses = torch.stack(poses_se3_list, 0)
    return poses


def se3_to_SE3_N(poses_wu):
    poses_se3_list = []
    for i in range(poses_wu.shape[0]):
        pose_se3 = se3_to_SE3(poses_wu[i])
        poses_se3_list.append(pose_se3)
    poses = torch.stack(poses_se3_list, 0)
    return poses


def se3_2_qt_parallel(wu):
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    # A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    # R = I + A * wx + B * wx @ wx
    V = I + B * wx + C * wx @ wx
    t = V @ u[..., None]
    q = exp_r2q_parallel(w)
    return q, t.squeeze(-1)


def SplineN_linear(start_pose, end_pose, poses_number, NUM, device=None):
    pose_time = poses_number / (NUM - 1)

    # parallel
    pos_0 = torch.where(pose_time == 0)
    pose_time[pos_0] = pose_time[pos_0] + 0.000001
    pos_1 = torch.where(pose_time == 1)
    pose_time[pos_1] = pose_time[pos_1] - 0.000001

    q_start, t_start = se3_2_qt_parallel(start_pose)
    q_end, t_end = se3_2_qt_parallel(end_pose)
    # sample t_vector
    t_t = (1 - pose_time)[..., None] * t_start + pose_time[..., None] * t_end

    # sample rotation_vector
    q_tau_0 = q_to_Q_parallel(q_to_q_conj_parallel(q_start)) @ q_end[..., None]
    r = pose_time[..., None] * log_q2r_parallel(q_tau_0.squeeze(-1))
    q_t_0 = exp_r2q_parallel(r)
    q_t = q_to_Q_parallel(q_start) @ q_t_0[..., None]

    # convert q&t to RT
    R = q_to_R_parallel(q_t.squeeze(dim=-1))
    t = t_t.unsqueeze(dim=-1)
    pose_spline = torch.cat([R, t], -1)

    poses = pose_spline.reshape([-1, 3, 4])

    return poses


def SplineN_cubic(pose0, pose1, pose2, pose3, poses_number, NUM):
    sample_time = poses_number / (NUM - 1)
    # parallel

    pos_0 = torch.where(sample_time == 0)
    sample_time[pos_0] = sample_time[pos_0] + 0.000001
    pos_1 = torch.where(sample_time == 1)
    sample_time[pos_1] = sample_time[pos_1] - 0.000001

    sample_time = sample_time.unsqueeze(-1)

    q0, t0 = se3_2_qt_parallel(pose0)
    q1, t1 = se3_2_qt_parallel(pose1)
    q2, t2 = se3_2_qt_parallel(pose2)
    q3, t3 = se3_2_qt_parallel(pose3)

    u = sample_time
    uu = sample_time**2
    uuu = sample_time**3
    one_over_six = 1.0 / 6.0
    half_one = 0.5

    # t
    coeff0 = one_over_six - half_one * u + half_one * uu - one_over_six * uuu
    coeff1 = 4 * one_over_six - uu + half_one * uuu
    coeff2 = one_over_six + half_one * u + half_one * uu - half_one * uuu
    coeff3 = one_over_six * uuu

    # spline t
    t_t = coeff0 * t0 + coeff1 * t1 + coeff2 * t2 + coeff3 * t3

    # R
    coeff1_r = 5 * one_over_six + half_one * u - half_one * uu + one_over_six * uuu
    coeff2_r = one_over_six + half_one * u + half_one * uu - 2 * one_over_six * uuu
    coeff3_r = one_over_six * uuu

    # spline R
    q_01 = q_to_Q_parallel(q_to_q_conj_parallel(q0)) @ q1[..., None]  # [1]
    q_12 = q_to_Q_parallel(q_to_q_conj_parallel(q1)) @ q2[..., None]  # [2]
    q_23 = q_to_Q_parallel(q_to_q_conj_parallel(q2)) @ q3[..., None]  # [3]

    r_01 = log_q2r_parallel(q_01.squeeze(-1)) * coeff1_r  # [4]
    r_12 = log_q2r_parallel(q_12.squeeze(-1)) * coeff2_r  # [5]
    r_23 = log_q2r_parallel(q_23.squeeze(-1)) * coeff3_r  # [6]

    q_t_0 = exp_r2q_parallel(r_01)  # [7]
    q_t_1 = exp_r2q_parallel(r_12)  # [8]
    q_t_2 = exp_r2q_parallel(r_23)  # [9]

    q_product1 = q_to_Q_parallel(q_t_1) @ q_t_2[..., None]  # [10]
    q_product2 = q_to_Q_parallel(q_t_0) @ q_product1  # [10]
    q_t = q_to_Q_parallel(q0) @ q_product2  # [10]

    R = q_to_R_parallel(q_t.squeeze(-1))
    t = t_t.unsqueeze(dim=-1)

    pose_spline = torch.cat([R, t], -1)

    poses = pose_spline.reshape([-1, 3, 4])

    return poses
