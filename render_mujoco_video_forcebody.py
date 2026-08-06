import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

xml_path = Path(os.environ["OUT"]) / "skel_humanoid.xml"
video_path = Path(os.environ["OUT"]) / "skel_body_static.mp4"

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Place the free-root model upright.
mujoco.mj_resetData(model, data)

data.qpos[0:3] = [0.0, 0.0, 1.0]

# +90 degrees around X: Y-up SKEL -> Z-up MuJoCo.
s = np.sqrt(0.5)
data.qpos[3:7] = [s, s, 0.0, 0.0]

mujoco.mj_forward(model, data)

width = 960
height = 720
model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
model.vis.global_.offheight = max(model.vis.global_.offheight, height)

renderer = mujoco.Renderer(model, width=width, height=height)

camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.lookat[:] = [0.0, 0.0, 1.0]
camera.distance = 2.4
camera.azimuth = 135.0
camera.elevation = -5.0

writer = imageio.get_writer(
    str(video_path),
    fps=30,
    codec="libx264",
    quality=8,
    macro_block_size=None,
)

try:
    num_frames = 180

    for frame_index in range(num_frames):
        # Rotate the camera once around the static body.
        camera.azimuth = 360.0 * frame_index / num_frames

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        writer.append_data(frame)
finally:
    writer.close()
    renderer.close()

print("Saved:", video_path)
