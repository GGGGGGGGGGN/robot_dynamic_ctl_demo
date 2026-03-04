import numpy as np
import time
import math

# 导入你的神兵利器
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.controllers import JointPDController, ComputedTorqueController

def main():
    # 1. 配置文件路径 (请根据你的实际路径调整)
    xml_path = "rm_control/assets/franka_emika_panda/scene.xml"  
    urdf_path = "rm_control/assets/panda_description/urdf/panda.urdf"

    print("启动仿真引擎...")
    sim = SimInterface(xml_path=xml_path, render=True, dt=0.001)
    sim.set_control_mode("torque")
    dyn_model = PinocchioDynamics(urdf_path=urdf_path)

    # 2. 设定相同的 PD 增益 (为了公平对比)
    # 故意不用特别大的增益，以突显前馈的威力
    kp = np.array([300, 300, 300, 300, 100, 100, 100])
    kd = np.array([30, 30, 30, 30, 10, 10, 10])

    # 3. 实例化两个控制器
    # 控制器 A: 仅重力+科氏力补偿 (Stage 1 的终极形态)
    ctrl_pd_grav = JointPDController(kp=kp, kd=kd, pin_dyn=dyn_model)
    # 控制器 B: 完整的计算力矩控制 (Stage 2 的主角)
    kp_ctc = np.array([900, 900, 900, 900, 400, 400, 400])
    kd_ctc = np.array([60, 60, 60, 60, 40, 40, 40])
    ctrl_ctc = ComputedTorqueController(kp=kp_ctc, kd=kd_ctc, pin_dyn=dyn_model)

    # 4. 轨迹生成器参数：让关节 1 (基座旋转) 和 关节 4 (肘部上下) 做高频正弦运动
    q_init = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    amplitude = np.array([0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0]) # 摆动幅度
    freq = 1.0 # 1 Hz (对于机械臂来说非常快了！)

    def get_trajectory(t):
        """生成目标位置、速度和加速度"""
        w = 2 * math.pi * freq
        q_ref = q_init + amplitude * math.sin(w * t)
        dq_ref = amplitude * w * math.cos(w * t)
        ddq_ref = -amplitude * w**2 * math.sin(w * t)
        return q_ref, dq_ref, ddq_ref

    import mujoco
    # 强制将 MuJoCo 的内部状态设置为我们的目标初始点
    sim.data.qpos[:7] = q_init
    sim.data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(sim.model, sim.data) # 更新运动学

    # =====================================================================
    # 测试 A：PD + 重力补偿 (高速动态跟踪)
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 测试 A: PD + 重力补偿 (忽略惯量 M)")
    print("="*50)
    print("观察现象：机械臂运动严重滞后，转折点处有巨大的‘甩飞’误差！")
    
    max_err_pd = 0.0
    for i in range(4000): # 运行 4 秒
        if not sim.is_alive(): break
        t = i * 0.001
        
        q, dq = sim.get_state()
        q_ref, dq_ref, ddq_ref = get_trajectory(t)
        
        # 故意不传 ddq_ref 给 PD，因为它不用 M，传了也没用
        tau = ctrl_pd_grav.update(q, dq, q_ref, dq_ref, np.zeros(7))
        
        sim.set_joint_torque(tau)
        sim.step()
        
        # 记录并打印关节 1 的误差
        err = abs(q_ref[0] - q[0])
        max_err_pd = max(max_err_pd, err)
        if i % 500 == 0:
            print(f"Time {t:.1f}s | 关节1 动态误差: {err:.4f} rad")

    time.sleep(1)
    sim.data.qpos[:7] = q_init
    sim.data.qvel[:7] = np.zeros(7)
    mujoco.mj_forward(sim.model, sim.data) # 更新运动学
    # =====================================================================
    # 测试 B：CTC 计算力矩控制 (全动力学降维打击)
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 测试 B: CTC 计算力矩控制 (引入 M(q) * ddq_ref)")
    print("="*50)
    print("观察现象：机械臂仿佛被铁轨锁死，指哪打哪，轨迹丝滑无比！")
    
    max_err_ctc = 0.0
    for i in range(4000): # 运行 4 秒
        if not sim.is_alive(): break
        t = i * 0.001
        
        q, dq = sim.get_state()
        q_ref, dq_ref, ddq_ref = get_trajectory(t)
        
        # CTC 需要完整的 ddq_ref！
        tau = ctrl_ctc.update(q, dq, q_ref, dq_ref, ddq_ref)
        
        sim.set_joint_torque(tau)
        sim.step()
        
        err = abs(q_ref[0] - q[0])
        max_err_ctc = max(max_err_ctc, err)
        if i % 500 == 0:
            print(f"Time {t:.1f}s | 关节1 动态误差: {err:.4f} rad")

    print("\n" + "="*50)
    print(f"📊 结果对比 (关节1 最大动态误差):")
    print(f"PD+重力补偿: {max_err_pd:.4f} rad")
    print(f"CTC 全动力学: {max_err_ctc:.4f} rad")
    print("="*50)

    print("\n测试完成，关闭渲染窗口退出...")
    while sim.is_alive():
        sim.step()
        time.sleep(0.01)

if __name__ == "__main__":
    main()