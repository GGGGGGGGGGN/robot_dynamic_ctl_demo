import numpy as np
import time
import sys
import os
import mujoco

sys.path.append(os.getcwd())

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_torque, get_model_path_urdf

def main():
    print("🚀 启动任务空间 (笛卡尔空间) 控制测试...")
    print("   原理: tau = J.T * (Kp * err_pos) + g(q)")

    # 1. 初始化
    xml_path = get_model_path_torque()
    urdf_path = get_model_path_urdf()
    sim = SimInterface(xml_path, render=True)
    
    mj_joints = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    # 🔥 必须指定末端，要控制谁，就传谁的名字
    TARGET_EE = "r_link6"
    pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=mj_joints, ee_name=TARGET_EE)

    # 2. 定义目标 (让右手悬停在空间某一点)
    # 我们先读取当前的初始位置，然后往上、往前加一点偏移
    q0, _ = sim.get_state()
    pin_dyn.update(q0, np.zeros(sim.nv))
    
    # 获取当前末端位置 (SE3 Object)
    # pin.updateFramePlacements 是 update() 内部调用的，但为了保险我们显式获取
    import pinocchio as pin
    pin.updateFramePlacements(pin_dyn.model, pin_dyn.data)
    init_pos = pin_dyn.data.oMf[pin_dyn.ee_id].translation.copy()
    
    # 设定目标：当前位置 向上 0.2m，向前 0.1m
    target_pos = init_pos + np.array([0.2, 0.0, 0.2]) 
    
    print(f"📍 初始位置: {init_pos}")
    print(f"🎯 目标位置: {target_pos}")

    # 3. 控制参数
    # 笛卡尔空间的刚度 (N/m)
    Kp_cartesian = 500.0  
    Kd_cartesian = 20.0   
    
    # 关节空间的阻尼 (防止零零散散的关节乱动)
    Kd_joint = 2.0

    sim_start = sim.get_time()
    
    while sim.is_alive():
        # A. 状态更新
        q, dq = sim.get_state()
        pin_dyn.update(q, dq)
        
        # B. 获取当前末端位置和雅可比
        # 这一步已经在 pin_dyn.update 里做了一部分，但我们需要 Frame 数据
        pin.updateFramePlacements(pin_dyn.model, pin_dyn.data)
        current_pos = pin_dyn.data.oMf[pin_dyn.ee_id].translation
        
        # 获取雅可比 (6 x nv)
        J = pin_dyn.get_jacobian()
        # 我们只控制位置 (前3行)，不管姿态 (后3行) -> 简化版点控制
        J_pos = J[:3, :] 
        
        # 获取末端线速度 v = J * dq
        current_vel = J_pos @ dq

        # C. 计算笛卡尔空间的虚拟力 F (弹簧阻尼模型)
        # F = Kp * (target - current) - Kd * current_vel
        pos_error = target_pos - current_pos
        F_des = Kp_cartesian * pos_error - Kd_cartesian * current_vel
        
        # D. 映射回关节力矩 tau = J.T * F
        tau_task = J_pos.T @ F_des
        
        # E. 加上重力补偿 g(q) 和 关节阻尼
        M, h = pin_dyn.get_dynamics() # h 包含重力
        tau_damp = -Kd_joint * dq     # 关节空间的微小阻尼，防止零空间(Nullspace)漂移
        
        tau_total = tau_task + h + tau_damp
        
        # F. 发送
        sim.set_whole_body_cmd(tau_total)
        sim.step()
        
        time.sleep(sim.dt)
        
        # 打印误差
        if sim.get_time() % 0.5 < sim.dt:
            err_norm = np.linalg.norm(pos_error)
            print(f"Error: {err_norm:.4f} m | F_z: {F_des[2]:.1f} N")

if __name__ == "__main__":
    main()