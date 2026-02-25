import numpy as np
import time
import math

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.controllers import PIDComputedTorqueController, MomentumObserverCTC
from rm_control.assets import get_model_path_xml, get_model_path_urdf
# 🔥 替换为正弦轨迹发生器
from rm_control.planning.trajectory_generator import SineTrajectory
from rm_control.utils.plotter import plot_tracking_comparison

# =====================================================================
# 工具函数 1：量化评估器 (输出极其专业的 Markdown 表格)
# =====================================================================
def evaluate_and_print_table(name, err_history):
    """计算并打印 7 个关节的 Max Error 和 RMSE"""
    max_errs = np.max(np.abs(err_history), axis=0)
    rms_errs = np.sqrt(np.mean(err_history**2, axis=0))
    
    print(f"\n📊 [{name}] 全关节跟踪性能评估:")
    print(f"| Joint | Max Error (rad) | RMSE (rad)    |")
    print(f"|-------|-----------------|---------------|")
    for i in range(7):
        print(f"| J{i+1:<4}| {max_errs[i]:<15.4f}| {rms_errs[i]:<13.4f}|")
    
    # 提取承受重力极大的关节 2 和 4 的平均表现作为总分参考
    critical_score = (max_errs[1] + max_errs[3]) / 2
    print(f"💡 关键受力关节 (J2, J4) 平均最大误差: {critical_score:.4f} rad\n")

# =====================================================================
# 工具函数 2：标准测试循环
# =====================================================================
def run_test_loop(sim, controller, traj, duration=4.0):
    t_list, q_ref_list, q_list, dq_ref_list, dq_list, err_list = [], [], [], [], [], []
    
    steps = int(duration / sim.dt)
    for i in range(steps):
        if not sim.is_alive(): break
        t = i * sim.dt
        
        q, dq = sim.get_state()
        q_ref, dq_ref, ddq_ref = traj.get_state(t)
        
        tau = controller.update(q, dq, q_ref, dq_ref, ddq_ref)
        sim.set_joint_torque(tau)
        sim.step()
        
        err = q_ref - q
        t_list.append(t)
        q_ref_list.append(q_ref)
        q_list.append(q)
        dq_ref_list.append(dq_ref)
        dq_list.append(dq)
        err_list.append(err)
        
    return (t_list, 
            np.array(q_ref_list), np.array(q_list), 
            np.array(dq_ref_list), np.array(dq_list), 
            np.array(err_list))

# =====================================================================
# 主程序
# =====================================================================
def main():
    print("启动仿真引擎...")
    
    # 1. 实例化环境并挂载 5kg 致命负载
    sim = SimInterface(
        xml_path=get_model_path_xml(), 
        render=True, # 🔥 建议打开渲染，欣赏狂暴甩臂
        dt=0.001,
        payload_mass=5.0,              
        payload_offset=[0.0, 0.0, 0.21], 
        payload_size=0.02
    )
    sim.set_control_mode("torque")
    dyn_model = PinocchioDynamics(urdf_path=get_model_path_urdf())

    # =====================================================================
    # 2. 控制器巅峰对决
    # =====================================================================
    ctrl_pid_ctc = PIDComputedTorqueController(
        kp=np.array([900, 900, 900, 900, 400, 400, 400]), 
        kd=np.array([60, 60, 60, 60, 40, 40, 40]), 
        ki=np.array([20, 20, 20, 20, 20, 20, 20]), 
        pin_dyn=dyn_model, 
        kv_fric=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        kc_fric=np.zeros(7),
        integral_limit=2.0 
    )

    ctrl_mob_ctc = MomentumObserverCTC(
        kp=np.array([900, 900, 900, 900, 400, 400, 400]), 
        kd=np.array([60, 60, 60, 60, 40, 40, 40]), 
        # ko=np.array([50, 50, 50, 50, 50, 50, 50]), 
        ko = np.array([100, 100, 100, 100, 100, 100, 100]), # 🔥 加强观测器增益，提升动态响应速度
        pin_dyn=dyn_model
    )
    
    # =====================================================================
    # 3. 🔥 实例化狂暴正弦轨迹
    # =====================================================================
    q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    
    # 保持振幅 0.2 rad
    amplitude = np.array([0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0])
    
    # 🔥 频率拉升至 0.8 Hz (1.25秒甩一个来回，速度极快！)
    frequency = 1
    
    traj = SineTrajectory(q_init=q_start, amplitude=amplitude, freq=frequency, duration=5.0)
    
    print("\n" + "="*50)
    print(f"📍 轨迹配置信息: {traj.__class__.__name__}")
    print(f"   - 初始姿态: {np.round(q_start, 3)}")
    print(f"   - 振幅: {amplitude}")
    print(f"   - 频率: {frequency} Hz")
    print("="*50 + "\n")

    # =====================================================================
    # 4. 运行对比测试
    # =====================================================================
    print("🚀 测试 A: PID-CTC 控制器 (在高速动态中被离心力暴打)")
    sim.set_robot_state(q=q_start, dq=np.zeros(7)) 
    ctrl_pid_ctc.error_sum = np.zeros(7) 
    t_pid, q_ref_pid, q_pid, dq_ref_pid, dq_pid, err_pid = run_test_loop(sim, ctrl_pid_ctc, traj, duration=traj.duration)
    evaluate_and_print_table("PID-CTC (Dynamic)", err_pid)

    time.sleep(1)

    print("🚀 测试 B: MOB-CTC 控制器 (动态对账神探)")
    sim.set_robot_state(q=q_start, dq=np.zeros(7))
    ctrl_mob_ctc.is_initialized = False 
    ctrl_mob_ctc.p_hat = np.zeros(7)
    ctrl_mob_ctc.last_tau_cmd = np.zeros(7)
    
    t_mob, q_ref_mob, q_mob, dq_ref_mob, dq_mob, err_mob = run_test_loop(sim, ctrl_mob_ctc, traj, duration=traj.duration)
    evaluate_and_print_table("MOB-CTC (Dynamic)", err_mob)

    # =====================================================================
    # 5. 画图展示！(继续对 Joint 4 进行处刑)
    # =====================================================================
    target_jnt_idx = 3 # J4 (肘部，受非线性动态力影响最剧烈的地方)
    
    plot_tracking_comparison(
        t_pid, 
        q_ref_pid[:, target_jnt_idx], dq_ref_pid[:, target_jnt_idx], 
        q_pid[:, target_jnt_idx], dq_pid[:, target_jnt_idx], err_pid[:, target_jnt_idx], "PID-CTC", "b-",
        q_mob[:, target_jnt_idx], dq_mob[:, target_jnt_idx], err_mob[:, target_jnt_idx], "MOB-CTC", "r-",
        joint_idx=target_jnt_idx + 1, 
        title_suffix="(5kg Payload: Dynamic Sine Wave Tracking)"
    )

    print("\n测试完成，关闭渲染窗口退出...")
    if sim.viewer is not None:
        sim.viewer.close()

if __name__ == "__main__":
    main()