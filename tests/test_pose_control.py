import numpy as np
import time
import sys
import os
import mujoco
import pinocchio as pin # 必须显式引入

sys.path.append(os.getcwd())

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_torque, get_model_path_urdf

def main():
    print("🚀 启动六维全姿态控制 (Pose Control)...")
    print("   目标: 手腕固定在空间中，且保持手掌水平，不准乱转！")

    # 1. 初始化
    xml_path = get_model_path_torque()
    urdf_path = get_model_path_urdf()
    sim = SimInterface(xml_path,  render=True)
    
    mj_joints = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    TARGET_EE = "r_link6"
    pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=mj_joints, ee_name=TARGET_EE)

    # 2. 设定目标姿态 (SE3)
    q0, _ = sim.get_state()
    pin_dyn.update(q0, np.zeros(sim.nv))
    pin.updateFramePlacements(pin_dyn.model, pin_dyn.data)
    
    # 获取初始位姿
    start_SE3 = pin_dyn.data.oMf[pin_dyn.ee_id].copy()
    
    # 设定目标: 
    # 位置: 向上 0.1m, 向前 0.1m
    # 姿态: 保持初始姿态不变 (或者你可以手动旋转它)
    target_SE3 = start_SE3.copy()
    target_SE3.translation += np.array([0.1, 0.0, 0.1])
    
    print(f"📍 初始位姿:\n{start_SE3}")
    
    # 3. 控制参数
    Kp_pos = 500.0   # 位置刚度
    Kp_ori = 100.0   # 姿态刚度 (通常给小一点)
    Kd_pos = 20.0
    Kd_ori = 5.0
    Kd_joint = 2.0   # 关节阻尼

    while sim.is_alive():
        # A. 状态更新
        q, dq = sim.get_state()
        pin_dyn.update(q, dq)
        
        # B. 获取当前末端位姿 (SE3) 和 雅可比 (6xN)
        # 注意: 必须显式调用 updateFramePlacements
        pin.updateFramePlacements(pin_dyn.model, pin_dyn.data)
        current_SE3 = pin_dyn.data.oMf[pin_dyn.ee_id]
        
        # 获取完整 6D 雅可比 (Local World Aligned)
        J = pin_dyn.get_jacobian() 
        
        # C. 计算误差
        # 1. 位置误差 (简单向量相减)
        err_pos = target_SE3.translation - current_SE3.translation
        
        # 2. 姿态误差 (关键点!)
        # 也就是计算: R_des * R_curr.T 的旋转向量
        # Pinocchio 提供了 log3 函数可以直接算这两个旋转矩阵的差异向量 (omega)
        # err_ori = log3(R_current.T @ R_target) (在局部坐标系)
        # 或者更简单的：pin.log6(current_SE3.actInv(target_SE3)).angular
        # 这里为了直观，我们用一种近似方法：
        R_err = target_SE3.rotation @ current_SE3.rotation.T
        err_ori = pin.log3(R_err) # 将旋转矩阵转为旋转向量 (3维)

        # D. 计算笛卡尔空间虚拟力 (6维: 3力 + 3力矩)
        # 速度 v_cartesian = J @ dq
        v_cartesian = J @ dq
        v_lin = v_cartesian[:3]
        v_ang = v_cartesian[3:]
        
        # F = Kp * err - Kd * v
        F_lin = Kp_pos * err_pos - Kd_pos * v_lin
        F_ang = Kp_ori * err_ori - Kd_ori * v_ang
        
        F_6d = np.hstack([F_lin, F_ang]) # 拼成 6维 向量
        
        # E. 映射回关节力矩 tau = J.T @ F_6d
        tau_task = J.T @ F_6d
        
        # F. 加上动力学前馈
        M, h = pin_dyn.get_dynamics()
        tau_damp = -Kd_joint * dq
        
        tau_total = tau_task + h + tau_damp
        
        sim.set_whole_body_cmd(tau_total)
        sim.step()
        
        # 渲染
        time.sleep(sim.dt)
        
        if sim.get_time() % 0.5 < sim.dt:
            e_p = np.linalg.norm(err_pos)
            e_o = np.linalg.norm(err_ori)
            print(f"Pos Err: {e_p:.4f} m | Ori Err: {e_o:.4f} rad")

if __name__ == "__main__":
    main()