import mujoco
import mujoco.viewer
import numpy as np
import time

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.02


def get_frame_jacobian(model, data, site_id) -> np.ndarray:
    jac = np.empty((6, model.nv))
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)

    # MuJoCo jacobians have a frame of reference centered at the local frame but
    # aligned with the world frame. To obtain a jacobian expressed in the local
    # frame, aka body jacobian, we need to left-multiply by A[T_fw].
    R_wf = data.site(site_id).xmat
    R_wf = R_wf.reshape(3, 3)
    R_fw = R_wf.T
    A_fw = np.block([[R_fw, np.zeros((3, 3))], [np.zeros((3, 3)), R_fw]])
    jac = A_fw @ jac

    return jac


def get_transform_frame_to_world(model, data, frame_id) -> np.ndarray:
    xpos = data.site(frame_id).xpos
    xmat = data.site(frame_id).xmat
    T = np.eye(4)
    T[:3, 3] = xpos
    R = xmat.reshape(3, 3)
    T[:3, :3] = R
    return T


def get_transform(
    model,
    data,
    source_id: int,
    dest_id: int,
):
    transform_source_to_world = get_transform_frame_to_world(model, data, source_id)
    transform_dest_to_world = get_transform_frame_to_world(model, data, dest_id)
    transform_world_to_dest = se3_inverse(transform_dest_to_world)
    return transform_world_to_dest @ transform_source_to_world


def skew(x: np.ndarray) -> np.ndarray:
    assert x.shape == (3,)
    wx, wy, wz = x
    return np.array(
        [
            [0.0, -wz, wy],
            [wz, 0.0, -wx],
            [-wy, wx, 0.0],
        ],
        dtype=x.dtype,
    )


def _getQ(c) -> np.ndarray:
    theta_sq = c[3:] @ c[3:]
    A = 0.5
    if theta_sq < 1e-5:
        B = (1.0 / 6.0) + (1.0 / 120.0) * theta_sq
        C = -(1.0 / 24.0) + (1.0 / 720.0) * theta_sq
        D = -(1.0 / 60.0)
    else:
        theta = np.sqrt(theta_sq)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        B = (theta - sin_theta) / (theta_sq * theta)
        C = (1.0 - theta_sq / 2.0 - cos_theta) / (theta_sq * theta_sq)
        D = ((2) * theta - (3) * sin_theta + theta * cos_theta) / (
            (2) * theta_sq * theta_sq * theta
        )
    V = skew(c[:3])
    W = skew(c[3:])
    VW = V @ W
    WV = VW.T
    WVW = WV @ W
    VWW = VW @ W
    return (
        +A * V
        + B * (WV + VW + WVW)
        - C * (VWW - VWW.T - 3 * WVW)
        + D * (WVW @ W + W @ WVW)
    )


def so3_ljacinv(other: np.ndarray) -> np.ndarray:
    theta = np.sqrt(other @ other)
    use_taylor = theta < 1e-5
    if use_taylor:
        t2 = theta**2
        A = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
    else:
        A = (1.0 / theta**2) * (
            1.0 - (theta * np.sin(theta) / (2.0 * (1.0 - np.cos(theta))))
        )
    skew_other = skew(other)
    return np.eye(3) - 0.5 * skew_other + A * (skew_other @ skew_other)


def se3_inverse(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    translation = T[:3, 3]
    R_inv = R.T
    return np.block(
        [
            [R_inv, -R_inv @ translation.reshape(3, 1)],
            [np.zeros((1, 3)), np.array([[1.0]])],
        ]
    )


def se3_ljacinv(other: np.ndarray) -> np.ndarray:
    theta = other[3:]
    if theta @ theta < 1e-5:
        return np.eye(6)
    Q = _getQ(other)
    J_inv = so3_ljacinv(theta)
    O = np.zeros((3, 3))
    return np.block([[J_inv, -J_inv @ Q @ J_inv], [O, J_inv]])


def se3_rjacinv(T) -> np.ndarray:
    return se3_ljacinv(-T)


def so3_log(wxyz: np.ndarray) -> np.ndarray:
    w = wxyz[0]
    norm_sq = wxyz[1:] @ wxyz[1:]
    use_taylor = norm_sq < 1e-5
    norm_safe = 1.0 if use_taylor else np.sqrt(norm_sq)
    w_safe = w if use_taylor else 1.0
    atan_n_over_w = np.arctan2(-norm_safe if w < 0 else norm_safe, abs(w))
    if use_taylor:
        atan_factor = 2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe**3
    else:
        if abs(w) < 1e-5:
            scl = 1.0 if w > 0.0 else -1.0
            atan_factor = scl * np.pi / norm_safe
        else:
            atan_factor = 2.0 * atan_n_over_w / norm_safe
    return atan_factor * wxyz[1:]


def se3_log(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    translation = T[:3, 3]
    wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(wxyz, R.ravel())
    omega = so3_log(wxyz)
    theta_squared = omega @ omega
    use_taylor = theta_squared < 1e-5
    skew_omega = skew(omega)
    theta_squared_safe = 1.0 if use_taylor else theta_squared
    theta_safe = np.sqrt(theta_squared_safe)
    half_theta_safe = 0.5 * theta_safe
    skew_omega_norm = skew_omega @ skew_omega
    if use_taylor:
        V_inv = np.eye(3, dtype=np.float64) - 0.5 * skew_omega + skew_omega_norm / 12.0
    else:
        V_inv = (
            np.eye(3, dtype=np.float64)
            - 0.5 * skew_omega
            + (
                1.0
                - theta_safe * np.cos(half_theta_safe) / (2.0 * np.sin(half_theta_safe))
            )
            / theta_squared_safe
            * skew_omega_norm
        )
    return np.concatenate([V_inv @ translation, omega])


def se3_jlog(T: np.ndarray) -> np.ndarray:
    return se3_rjacinv(se3_log(T))


def se3_adjoint(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return np.block(
        [
            [R, skew(T[:3, 3]) @ R],
            [np.zeros((3, 3), dtype=np.float64), R],
        ]
    )


def get_relative_jacobian(
    model, data, frame_id, root_id, transform_target_to_root
) -> np.ndarray:
    jacobian_frame_in_frame = get_frame_jacobian(model, data, frame_id)
    jacobian_root_in_root = get_frame_jacobian(model, data, root_id)

    transform_frame_to_root = get_transform(model, data, frame_id, root_id)
    transform_root_to_target = se3_inverse(transform_target_to_root)
    transform_frame_to_target = transform_root_to_target @ transform_frame_to_root

    transform_root_to_frame = se3_inverse(transform_frame_to_root)
    adjoint_root_to_frame = se3_adjoint(transform_root_to_frame)

    return se3_jlog(transform_frame_to_target) @ (
        jacobian_frame_in_frame - adjoint_root_to_frame @ jacobian_root_in_root
    )


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("unitree_g1/scene.xml")
    data = mujoco.MjData(model)

    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "right_palm"
    site_id = model.site(site_name).id
    base_name = "imu_in_torso"
    base_id = model.site(base_name).id
    mocap_name = "right_palm_target"
    mocap_id = model.site(mocap_name).id

    # sensor
    torso_vel_sensor_name = "imu-torso-linear-velocity"
    torso_vel_sensor_id = model.sensor(torso_vel_sensor_name).id
    right_palm_vel_sensor_name = "imu-right-palm-linear-velocity"
    right_palm_vel_sensor_id = model.sensor(right_palm_vel_sensor_name).id

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML le. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "stand"
    key_id = model.key(key_name).id
    q0 = model.key(key_name).qpos

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        base_vel = np.zeros(3)
        while viewer.is_running():
            step_start = time.time()

            transform_target_to_root = get_transform(model, data, mocap_id, base_id)
            jac_site_root = get_relative_jacobian(
                model, data, site_id, base_id, transform_target_to_root
            )
            transform_site_to_root = get_transform(model, data, site_id, base_id)
            transform_site_to_target = (
                se3_inverse(transform_target_to_root) @ transform_site_to_root
            )
            # transform_site_to_target = get_transform(model, data, site_id, mocap_id)
            error = se3_log(transform_site_to_target)
            # regularize jacobian
            dq = -np.linalg.pinv(jac_site_root) @ error * 10.0

            # perturb base velocity
            base_vel += np.random.uniform(-0.5, 0.5, 3) * dt
            base_vel += 1.0 * (q0[:3] - data.qpos[:3])
            base_vel = np.clip(base_vel, -3.0, 3.0)
            dq[:3] += base_vel
            dq[3:6] += np.random.randn(3) * 0.0

            # Integrate joint velocities to obtain joint positions.
            mujoco.mj_integratePos(model, data.qpos, dq, dt)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_camlight(model, data)

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
