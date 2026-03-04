import numpy as np
import time
import math
import matplotlib.pyplot as plt  # 引入画图神器

# 导入你的神兵利器
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.controllers import JointPDController, ComputedTorqueController, ComputedTorqueControllerWithFriction

def main():
    # 1. 配置文件路径 (请根据你的实际路径调整)
    xml_path = "rm_control/assets/franka_emika_panda/scene.xml"  
    urdf_path = "rm_control/assets/panda_description/urdf/panda.urdf"

    print("启动仿真引擎...")
    sim = SimInterface(xml_path=xml_path, render=True, dt=0.001)
    sim.set_control_mode("torque")
    dyn_model = PinocchioDynamics(urdf_path=urdf_path)

    # 2. 设定 PD 增益
    kp = np.array([300, 300, 300, 300, 100, 100, 100])
    kd = np.array([30, 30, 30, 30, 10, 10, 10])

    # 3. 实例化两个控制器
    ctrl_pd_grav = JointPDController(kp=kp, kd=kd, pin_dyn=dyn_model)
    
    kp_ctc = np.array([900, 900, 900, 900, 400, 400, 400])
    kd_ctc = np.array([60, 60, 60, 60, 40, 40, 40])
    exact_kv_fric = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    exact_kc_fric = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    ctrl_ctc = ComputedTorqueControllerWithFriction(
        kp=kp_ctc, 
        kd=kd_ctc, 
        pin_dyn=dyn_model, 
        kv_fric=exact_kv_fric,
        kc_fric=exact_kc_fric
    )
    
    # 4. 轨迹生成器参数
    q_init = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    amplitude = np.array([0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0]) 
    freq = 1.0 

    def get_trajectory(t):
        w = 2 * math.pi * freq
        q_ref = q_init + amplitude * math.sin(w * t)
        dq_ref = amplitude * w * math.cos(w * t)
        ddq_ref = -amplitude * w**2 * math.sin(w * t)
        return q_ref, dq_ref, ddq_ref

    import mujoco
    sim.data.qpos[:7] = q_init
    sim.data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(sim.model, sim.data)

    # =====================================================================
    # 准备数据记录容器 (针对关节 1)
    # =====================================================================
    t_pd, q_ref_pd, q_pd, dq_ref_pd, dq_pd, err_pd = [], [], [], [], [], []
    t_ctc, q_ref_ctc, q_ctc, dq_ref_ctc, dq_ctc, err_ctc = [], [], [], [], [], []

    # =====================================================================
    # 测试 A：PD + 重力补偿
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 测试 A: PD + 重力补偿 (忽略惯量 M)")
    print("="*50)
    
    max_err_pd = 0.0
    for i in range(4000):
        if not sim.is_alive(): break
        t = i * 0.001
        
        q, dq = sim.get_state()
        q_ref, dq_ref, ddq_ref = get_trajectory(t)
        
        tau = ctrl_pd_grav.update(q, dq, q_ref, dq_ref, np.zeros(7))
        sim.set_joint_torque(tau)
        sim.step()
        
        err = q_ref[0] - q[0]
        max_err_pd = max(max_err_pd, abs(err))
        
        # 记录数据
        t_pd.append(t)
        q_ref_pd.append(q_ref[0])
        q_pd.append(q[0])
        dq_ref_pd.append(dq_ref[0])
        dq_pd.append(dq[0])
        err_pd.append(err)

    time.sleep(1)
    sim.data.qpos[:7] = q_init
    sim.data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(sim.model, sim.data) 
    
    # =====================================================================
    # 测试 B：CTC 计算力矩控制
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 测试 B: CTC 计算力矩控制 (引入 M(q) * ddq_ref)")
    print("="*50)
    
    max_err_ctc = 0.0
    for i in range(4000):
        if not sim.is_alive(): break
        t = i * 0.001
        
        q, dq = sim.get_state()
        q_ref, dq_ref, ddq_ref = get_trajectory(t)
        
        tau = ctrl_ctc.update(q, dq, q_ref, dq_ref, ddq_ref)
        sim.set_joint_torque(tau)
        sim.step()
        
        err = q_ref[0] - q[0]
        max_err_ctc = max(max_err_ctc, abs(err))
        
        # 记录数据
        t_ctc.append(t)
        q_ref_ctc.append(q_ref[0])
        q_ctc.append(q[0])
        dq_ref_ctc.append(dq_ref[0])
        dq_ctc.append(dq[0])
        err_ctc.append(err)

    print("\n" + "="*50)
    print(f"📊 结果对比 (关节1 最大动态误差):")
    print(f"PD+重力补偿: {max_err_pd:.4f} rad")
    print(f"CTC 全动力学: {max_err_ctc:.4f} rad")
    print("="*50)

    # =====================================================================
    # 开始画图！
    # =====================================================================
    print("📊 正在绘制轨迹分析图...")
    plt.figure(figsize=(12, 10))

    # 子图 1：位置跟踪
    plt.subplot(3, 1, 1)
    plt.plot(t_pd, q_ref_pd, 'k--', linewidth=2, label='Target Position')
    plt.plot(t_pd, q_pd, 'b-', alpha=0.7, label='PD Control')
    plt.plot(t_ctc, q_ctc, 'r-', alpha=0.7, label='CTC Control')
    plt.ylabel('Position (rad)', fontsize=12)
    plt.title('Joint 1 Position Tracking', fontsize=14)
    plt.legend()
    plt.grid(True)

    # 子图 2：速度跟踪 (🚨 重点观察 t=0 处！)
    plt.subplot(3, 1, 2)
    plt.plot(t_pd, dq_ref_pd, 'k--', linewidth=2, label='Target Velocity')
    plt.plot(t_pd, dq_pd, 'b-', alpha=0.7, label='PD Control')
    plt.plot(t_ctc, dq_ctc, 'r-', alpha=0.7, label='CTC Control')
    plt.ylabel('Velocity (rad/s)', fontsize=12)
    plt.title('Joint 1 Velocity Tracking (Notice the Velocity Step at t=0!)', fontsize=14)
    plt.legend()
    plt.grid(True)

    # 子图 3：跟踪误差
    plt.subplot(3, 1, 3)
    plt.plot(t_pd, err_pd, 'b-', alpha=0.7, label='PD Error')
    plt.plot(t_ctc, err_ctc, 'r-', alpha=0.7, label='CTC Error')
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Error (rad)', fontsize=12)
    plt.title('Joint 1 Tracking Error (Notice the huge spike for CTC at t=0)', fontsize=14)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\n测试完成，关闭渲染窗口退出...")
    if sim.viewer is not None:
        sim.viewer.close()

if __name__ == "__main__":
    main()