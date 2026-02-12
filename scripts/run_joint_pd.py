import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # 必须在导入 pyplot 之前设置，否则 mjpython 线程会报错
import matplotlib.pyplot as plt

# 引入你的库
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.joint_pd import JointPDController
from rm_control.assets import get_model_path_xml, get_model_path_urdf

def main():
    # ---------------------------------------------------------
    # 1. 环境初始化
    # ---------------------------------------------------------

    # 加载 MuJoCo
    xml_path = get_model_path_xml()
    sim = SimInterface(xml_path, render=True)

    # 🔥 关键步骤：清理 MuJoCo 的物理干扰 (Nuclear Option)
    # # 这样可以确保是一个纯粹的刚体，完全由我们的 PD 控制器接管
    # sim.model.jnt_stiffness[:] = 0   # 关掉关节弹簧
    # sim.model.dof_damping[:] = 0     # 关掉关节阻尼
    # sim.model.dof_armature[:] = 0    # 关掉电枢惯量
    
    # 设置为纯力矩模式 (Gain=1, Bias=0)
    sim.set_control_mode("torque") 
    
    # 加载 Pinocchio (用于计算重力项 h)
    urdf_path = get_model_path_urdf()
    pin_dyn = PinocchioDynamics(urdf_path, ee_name="panda_link7")

    print("✅ 环境初始化完成，物理参数已清理。")

    # ---------------------------------------------------------
    # 2. 配置控制器
    # ---------------------------------------------------------
    # 设定一个目标姿态 (Panda 经典的 Ready Pose)
    q_target = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    # 调节增益 (KP, KD)
    # 根部关节负载大，刚度给大点；末端关节负载小，刚度给小点
    kp = np.array([800, 800, 800, 800, 500, 400, 300])
    # 经验法则：Kd 通常取 Kp 开根号的 2倍左右 (临界阻尼附近)
    kd = np.array([40,  40,  40,  40,  20,  20,  10])
    
    controller = JointPDController(kp, kd, q_target)
    
    # 重置机器人状态到全零，让它从零开始运动到目标
    sim.reset()
    
    # ---------------------------------------------------------
    # 3. 运行控制循环
    # ---------------------------------------------------------
    total_time = 3.0  # 秒
    dt = sim.model.opt.timestep
    steps = int(total_time / dt)
    
    # 数据记录
    log_q = []
    log_tau = []
    log_time = []

    print(f"🚀 开始运动控制，目标: {q_target}")
    
    for i in range(steps):
        start_time = time.time()
        
        # A. 获取状态
        q, dq = sim.get_state()
        
        # B. 更新动力学模型
        pin_dyn.update(q, dq)
        M, h = pin_dyn.get_dynamics() # 获取重力+科氏力
        
        # C. 计算力矩 (PD + Gravity Comp)
        tau = controller.compute(q, dq, h)
        
        tau = tau.flatten()
        # D. 发送力矩
        sim.set_joint_torque(tau)
        
        # E. 物理步进
        sim.step()
        
        # 记录数据
        log_q.append(q.copy())
        log_tau.append(tau.copy())
        log_time.append(i * dt)
        
        # 保持实时性 (可选)
        # while time.time() - start_time < dt: pass

    print("🏁 运动结束，正在绘图...")
    sim.viewer.close() # 关闭仿真窗口

    # ---------------------------------------------------------
    # 4. 结果可视化
    # ---------------------------------------------------------
    log_q = np.array(log_q)
    log_time = np.array(log_time)
    
    plt.figure(figsize=(10, 6))
    
    # 只画前 4 个关节，避免太乱
    colors = ['r', 'g', 'b', 'orange']
    for j in range(4):
        plt.plot(log_time, log_q[:, j], label=f'Joint {j+1}', color=colors[j])
        plt.axhline(q_target[j], linestyle='--', color=colors[j], alpha=0.5)
    
    plt.title("Joint PD Control Response (First 4 Joints)")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (rad)")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('joint_pd_result.png')
    print("📊 绘图已保存至 joint_pd_result.png")

if __name__ == "__main__":
    main()