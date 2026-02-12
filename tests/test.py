import numpy as np
import time
import mujoco

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_xml, get_model_path_urdf

def main():
    # 1. 初始化
    arm_joints = ["panda_joint1", "panda_joint2", "panda_joint3", 
                  "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    
    # 渲染开启
    sim = SimInterface(get_model_path_xml(), active_joint_names=arm_joints, render=True)
    
    # 2. 加载动力学
    # 注意：这里不需要传 active_joint_names，因为我们已经在 Pinocchio 代码里改了默认值
    pin_dyn = PinocchioDynamics(get_model_path_urdf())

    # ---------------------------------------------------------
    # 🔥🔥🔥 核心诊断与修复区域 🔥🔥🔥
    # ---------------------------------------------------------
    print("\n🔍 [1] 检查电机初始状态...")
    print(f"    Actuator 0 Gain (Before): {sim.model.actuator_gainprm[0, 0]}")
    
    # 强制修正：无论之前 XML 怎么写的，这里强行改成“纯力矩模式”
    # 1. 增益设为 1.0 (这样 10Nm 的指令就是 10Nm 的力)
    sim.model.actuator_gainprm[:, 0] = 1.0 
    # 2. 偏置设为 0 (去掉原本的 P 控制弹簧)
    sim.model.actuator_biasprm[:, :] = 0
    # 3. 关掉阻尼 (为了测试纯重力悬停效果)
    sim.model.dof_damping[:] = 0
    # 4. 去掉力矩限制 (防止截断)
    sim.model.actuator_forcerange[:] = np.array([-1000, 1000])
    
    print(f"✅ [2] 电机参数已强制修正: Gain=1.0, Bias=0")
    print("-" * 50)

    # 3. 设置一个悬臂姿态 (伸出去，受重力最大)
    # 姿态: [0, -0.78, 0, -2.35, 0, 1.57, 0.78]
    q_home = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78])
    sim.data.qpos[:7] = q_home
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)

    print("🚀 开始重力补偿循环...")
    
    while True:
        start = time.time()
        
        # A. 获取状态
        q, dq = sim.get_state()
        
        # B. Pinocchio 计算重力
        pin_dyn.update(q, dq)
        M, h = pin_dyn.get_dynamics()
        
        # C. 构造力矩 (h 就是重力+科氏力，静止时就是纯重力)
        tau_cmd = h.flatten() # 拍扁，防止维度错误
        
        # D. 补全维度 (7 -> nu)
        tau_full = np.zeros(sim.model.nu)
        tau_full[:7] = tau_cmd
        
        # E. 发送力矩
        sim.set_joint_torque(tau_full)
        
        # F. 步进
        sim.step()
        
        # 🔍 [实时监控]
        # 每 50 步打印一次，看看计算出的力矩是不是 0
        if sim.data.time % 0.1 < 0.002: # 约每0.1秒打印一次
            # 监控第 2 个关节 (肩膀)，它受力最大
            print(f"Time: {sim.data.time:.1f}s | J2 Pos: {q[1]:.2f} | J2 Calc Torque: {tau_cmd[1]:.2f} Nm")
            
            # 如果算出来的力矩是 0，那就是 Pinocchio 加载错了
            if np.abs(tau_cmd[1]) < 0.01:
                print("❌ 警告：Pinocchio 算出的重力为 0！检查 URDF 是否有质量参数！")

        time.sleep(0.002)

if __name__ == "__main__":
    main()