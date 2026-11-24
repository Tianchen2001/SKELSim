import numpy as np, mujoco, sys
m = mujoco.MjModel.from_xml_path(sys.argv[1] if len(sys.argv) > 1 else 'out/skel_dynamics.xml'); d = mujoco.MjData(m)
kp, kd = 400.0, 40.0
m.dof_damping[6:] = kd; m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # damping inside the integrator
TR, TL = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ('talus_r', 'talus_l'))
W = m.body_mass.sum() * 9.81
r = m.jnt_range[1:]; q0 = np.clip(d.qpos[7:], r[:, 0] + 0.02, r[:, 1] - 0.02)
q0[(r[:, 0] <= 0) & (r[:, 1] >= 0)] = 0.0
d.qpos[7:] = q0; mujoco.mj_forward(m, d)
f = np.zeros(6)
for step in range(2500):
    d.ctrl[:] = np.clip(kp * (q0 - d.qpos[7:]), m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
    mujoco.mj_step(m, d)
    if step in (0, 49, 149, 249, 499, 999, 1499, 2499):
        tot = 0.0
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f); tot += f[0] * abs(d.contact[i].frame[2])
        tilt = np.degrees(2 * np.arccos(min(1.0, abs(d.qpos[3]))))
        fwd = 1000 * (d.subtree_com[1][0] - 0.5 * (d.xpos[TR][0] + d.xpos[TL][0]))
        err = np.degrees(np.abs(q0 - d.qpos[7:]).max())
        print(f't={d.time:5.2f}s pelvis z {d.qpos[2]:.3f}  tilt {tilt:5.1f} deg  contacts {d.ncon:2d}  vertical force/weight {tot / W:5.2f}  max joint error {err:4.1f} deg  com ahead of ankles {fwd:5.0f} mm')
