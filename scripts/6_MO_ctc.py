import numpy as np
import time
import math

# 导入你的神兵利器
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
# 🔥 确保引入了你新写的 MomentumObserverCTC
from rm_control.controllers.controllers import PIDComputedTorqueController, MomentumObserverCTC
from rm_control.assets import get_model_path_xml, get_model_path_urdf
from rm_control.planning.trajectory_generator import FixedTrajectory
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
    """运行仿真循环，并收集所有 7 个关节的数据"""
    t_list, q_ref_list, q_list, dq_ref_list, dq_list, err_list = [], [], [], [], [], []
    
    steps = int(duration / sim.dt)
    for i in range(steps):
        if not sim.is_alive(): break
        t = i * sim.dt
        
        q, dq = sim.get_state()
        q_ref, dq_ref, ddq_ref = traj.get_state(t)
        
        tau = controller.update(q, dq, q_ref, dq_ref, ddq_ref)
        # 注意：MomentumObserverCTC 返回的是 tau_cmd，如果之前你修改让它返回元组，这里要取 tau[0]，
        # 如果你按我上一条的建议只返回 tau_cmd，则直接使用 tau
        sim.set_joint_torque(tau)
        sim.step()
        
        # 记录全量数据
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
        render=False, 
        dt=0.001,
        payload_mass=5.0,              
        payload_offset=[0.0, 0.0, 0.21], # 顺着 Z 轴往外挂
        payload_size=0.02
    )
    sim.set_control_mode("torque")
    dyn_model = PinocchioDynamics(urdf_path=get_model_path_urdf())

    # =====================================================================
    # 2. 巅峰对决：PID-CTC vs MOB-CTC
    # =====================================================================
    # 控制器 A：带积分项的传统 CTC (慢吞吞的老黄牛)
    ctrl_pid_ctc = PIDComputedTorqueController(
        kp=np.array([900, 900, 900, 900, 400, 400, 400]), 
        kd=np.array([60, 60, 60, 60, 40, 40, 40]), 
        ki=np.array([20, 20, 20, 20, 20, 20, 20]), # 微弱积分
        pin_dyn=dyn_model, 
        kv_fric=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        kc_fric=np.zeros(7),
        integral_limit=2.0 
    )

    # 控制器 B：带 动量观测器 (MOB) 的 CTC (反应神速的刺客)
    ctrl_mob_ctc = MomentumObserverCTC(
        kp=np.array([900, 900, 900, 900, 400, 400, 400]), 
        kd=np.array([60, 60, 60, 60, 40, 40, 40]), 
        ko=np.array([50, 50, 50, 50, 50, 50, 50]), # 🔥 观测带宽 50Hz
        pin_dyn=dyn_model
    )
    
    # 3. 实例化保位轨迹
    q_target = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    traj = FixedTrajectory(q_target=q_target, duration=4.0)
    
    print("\n" + "="*50)
    print(f"📍 轨迹配置信息: {traj.__class__.__name__}")
    print(f"   - 目标姿态: {np.round(q_target, 3)}")
    print("="*50 + "\n")

    # =====================================================================
    # 4. 运行对比测试
    # =====================================================================
    print("🚀 测试 A: PID-CTC 控制器 (老黄牛积分)")
    sim.set_robot_state(q=q_target, dq=np.zeros(7)) 
    ctrl_pid_ctc.error_sum = np.zeros(7) # 清空历史积分
    t_pid, q_ref_pid, q_pid, dq_ref_pid, dq_pid, err_pid = run_test_loop(sim, ctrl_pid_ctc, traj, duration=traj.duration)
    evaluate_and_print_table("PID-CTC", err_pid)

    # 给物理引擎一点喘息时间
    time.sleep(1)

    print("🚀 测试 B: MOB-CTC 控制器 (动量观测器)")
    sim.set_robot_state(q=q_target, dq=np.zeros(7))
    # 清空 MOB 的历史状态，保证公正
    ctrl_mob_ctc.is_initialized = False 
    ctrl_mob_ctc.p_hat = np.zeros(7)
    ctrl_mob_ctc.last_tau_cmd = np.zeros(7)
    
    t_mob, q_ref_mob, q_mob, dq_ref_mob, dq_mob, err_mob = run_test_loop(sim, ctrl_mob_ctc, traj, duration=traj.duration)
    evaluate_and_print_table("MOB-CTC", err_mob)

    # =====================================================================
    # 5. 画图展示！(提取受重力影响最大的 Joint 2 进行处刑)
    # =====================================================================
    target_jnt_idx = 1 # J2
    
    plot_tracking_comparison(
        t_pid, 
        q_ref_pid[:, target_jnt_idx], dq_ref_pid[:, target_jnt_idx], 
        q_pid[:, target_jnt_idx], dq_pid[:, target_jnt_idx], err_pid[:, target_jnt_idx], "PID-CTC", "b-",
        q_mob[:, target_jnt_idx], dq_mob[:, target_jnt_idx], err_mob[:, target_jnt_idx], "MOB-CTC", "r-",
        joint_idx=target_jnt_idx + 1, 
        title_suffix="(5kg Payload: Integrator vs Momentum Observer)"
    )

    print("\n测试完成，关闭渲染窗口退出...")
    if sim.viewer is not None:
        sim.viewer.close()

if __name__ == "__main__":
    main()