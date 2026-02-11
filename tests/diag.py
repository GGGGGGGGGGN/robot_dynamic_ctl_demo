import mujoco
import pinocchio as pin
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_torque, get_model_path_urdf

# 1. 加载环境
xml_path = get_model_path_torque()
sim = SimInterface(xml_path, render=False)
mj_joints = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]

# 2. 加载 Pinocchio (带裁剪)
urdf_path = get_model_path_urdf()
pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=mj_joints, ee_name="r_link6")

# 3. 打印诊断信息
print("-" * 30)
print("🔍 模型对齐诊断")
print("-" * 30)

# A. 检查 MuJoCo 的基座位置
# 找到 base_link (或者 platform_link) 的 ID
try:
    # 你的 XML 里可能是 platform_link 或者 base_link，视具体固定了哪个而定
    # 假设你固定了 platform_link，那么它的位置就是 body pos
    body_name = "base_link" # 改成你 XML 里那个主要的基座 body 名字
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    print(f"MuJoCo '{body_name}' Pos: {sim.model.body_pos[bid]}")
    print(f"MuJoCo '{body_name}' Quat: {sim.model.body_quat[bid]}")
except:
    print("❌ MuJoCo 里没找到指定 Body，请检查名字。")

# B. 检查 Pinocchio 的基座位置 (即锁死后的 World Frame -> Base Frame)
# 更新一次运动学
q = np.zeros(pin_dyn.model.nq)
pin.forwardKinematics(pin_dyn.model, pin_dyn.data, q)
pin.updateFramePlacements(pin_dyn.model, pin_dyn.data)

# 找到基座在 Pinocchio 里的 Frame ID (通常是 1 或 2，因为 joint 0 是 universe)
# 我们打印前几个 Frame 看看
for i in range(min(5, pin_dyn.model.nframes)):
    f = pin_dyn.model.frames[i]
    print(f"Pinocchio Frame {i} ({f.name}):\n{pin_dyn.data.oMf[i]}")

print("-" * 30)