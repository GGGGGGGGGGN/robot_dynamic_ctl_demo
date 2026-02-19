import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rm_control.simulation.sim_interface import SimInterface
from rm_control.controllers.controllers import JointPDController
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_xml, get_model_path_urdf
from rm_control.planning.trajectory_generator import StepTrajectory
from rm_control.utils.benchmark import BenchmarkRunner # 复用我们之前的 Runner


def main():

    
    # 当前测试的关节 ID (0代表J1, 6代表J7)
    TEST_JOINT_ID = 3  # J4
    
    # 测试参数 (Kp=800, Kd=40 是比较硬的参数，适合大关节)
    kp_test = 800.0
    kd_test = 60.0
    
    # 构造参数数组
    kp = np.array([600.0] * 7)
    kd = np.array([30.0] * 7)
    kp[TEST_JOINT_ID] = kp_test
    kd[TEST_JOINT_ID] = kd_test
    
    print(f"🎯 测试 J{TEST_JOINT_ID+1} | 范围: -1.5 -> -1.0 | Kp={kp_test}, Kd={kd_test}")

    # ... (Sim 和 Pinocchio 初始化不变) ...
    xml_path = get_model_path_xml()
    urdf_path = get_model_path_urdf()
    
    sim = SimInterface(xml_path, dt=0.001, render=True) # 调参不需要看动画，看曲线就行
    pin_dyn = PinocchioDynamics(urdf_path)
    
    # 使用带重力补偿的控制器
    ctrl = JointPDController(kp, kd, pin_dyn)
    runner = BenchmarkRunner(sim, ctrl)
    
    # 🔥【关键修改】设定合法的起点和终点
    # J4 是肘部，我们在 -1.5 (弯曲) 到 -1.0 (稍直) 之间测试
    traj = StepTrajectory(TEST_JOINT_ID, start_val=-1.5, end_val=-1.0, step_time=0.2)
    
    # 运行仿真
    history = runner.run(traj, duration=1.5)
    
    # ==========================================
    # 📈 专画这一关节的图
    # ==========================================
    t = np.array(history["t"])
    q_real = np.array(history["q"])[:, TEST_JOINT_ID]
    q_ref = np.array(history["q_ref"])[:, TEST_JOINT_ID]
    tau = np.array(history["tau"])[:, TEST_JOINT_ID]
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    
    # 图1: 位置响应
    ax = axes[0]
    ax.plot(t, q_ref, 'r--', label="Target")
    ax.plot(t, q_real, 'b-', lw=2, label="Real")
    ax.set_title(f"Joint {TEST_JOINT_ID+1} Step Response (Kp={kp_test}, Kd={kd_test})")
    ax.set_ylabel("Angle (rad)")
    ax.grid(True)
    ax.legend()
    
    # 图2: 力矩输出
    ax = axes[1]
    ax.plot(t, tau, 'g-')
    ax.set_title("Torque")
    ax.set_ylabel("Nm")
    ax.grid(True)
    
    save_name = f"tune_J{TEST_JOINT_ID+1}_kp{int(kp_test)}_kd{int(kd_test)}.png"
    plt.savefig(save_name)
    print(f"✅ 结果已保存: {save_name} (请打开图片分析波形)")

if __name__ == "__main__":
    main()