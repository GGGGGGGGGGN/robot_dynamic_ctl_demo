import numpy as np
import time

# 导入你写好的神兵利器
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.controllers import JointPDController

def main():
    # 1. 配置文件路径 (请确保路径与你本地的 assets 对应)
    # 根据你的项目结构，通常是这个路径
    xml_path = "rm_control/assets/franka_emika_panda/scene.xml"  
    urdf_path = "rm_control/assets/panda_description/urdf/panda.urdf"

    print("启动仿真引擎...")
    # 2. 初始化 MuJoCo 仿真接口
    sim = SimInterface(xml_path=xml_path, render=True, dt=0.001)
    sim.set_control_mode("torque") # 切入纯力矩控制模式
    
    # 3. 初始化 Pinocchio 动力学模型 (此时仅作为阶段2的重力计算器使用)
    dyn_model = PinocchioDynamics(urdf_path=urdf_path)

    # 4. 设定一个“伸懒腰”的目标姿态 
    # 这种姿态下，关节 2 (肩) 和 关节 4 (肘) 承受的重力力矩极大
    q_ref = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    dq_ref = np.zeros(7)
    ddq_ref = np.zeros(7)

    # 5. 设定 PD 参数 (故意不设得太大，以暴露纯 PD 的缺陷)
    kp = np.array([200, 200, 200, 200, 50, 50, 50])
    kd = np.array([20, 20, 20, 20, 5, 5, 5])

    # =====================================================================
    # 阶段 1：纯 PD 控制 (盲人摸象)
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 阶段 1：纯 PD 控制 (关闭重力补偿)")
    print("="*50)
    
    # 注意这里 pin_dyn 传了 None，你的控制器里退化为 Pure_PD
    controller_pure_pd = JointPDController(kp=kp, kd=kd, pin_dyn=None)

    # 运行 3 秒 (3000步)
    for i in range(3000):
        if not sim.is_alive():
            break
            
        q, dq = sim.get_state()
        tau = controller_pure_pd.update(q, dq, q_ref, dq_ref, ddq_ref)
        
        sim.set_joint_torque(tau)
        sim.step()
        
        # 每 0.5 秒打印一次误差
        if i % 500 == 0:
            error = q_ref - q
            print(f"Time {i/1000:.1f}s | 关节2误差: {error[1]:.4f} rad | 关节4误差: {error[3]:.4f} rad")
            
    print("\n🤯 观察现象：机械臂根本举不起来，存在巨大的【稳态误差】！重力把机械臂往下拽，PD 只能靠误差来产生对抗力。")
    time.sleep(2) # 暂停2秒给你观察画面

    # =====================================================================
    # 阶段 2：重力补偿 PD (物理降维打击)
    # =====================================================================
    print("\n" + "="*50)
    print("🚀 阶段 2：重力补偿 PD (开启物理模型挂)")
    print("="*50)
    
    # 传入 dyn_model，此时 tau_ff = h 会被激活！
    controller_grav_comp = JointPDController(kp=kp, kd=kd, pin_dyn=dyn_model)

    # 再运行 3 秒
    for i in range(30000):
        if not sim.is_alive():
            break
            
        q, dq = sim.get_state()
        tau = controller_grav_comp.update(q, dq, q_ref, dq_ref, ddq_ref)
        
        sim.set_joint_torque(tau)
        sim.step()
        
        if i % 500 == 0:
            error = q_ref - q
            print(f"Time {(i+3000)/1000:.1f}s | 关节2误差: {error[1]:.4f} rad | 关节4误差: {error[3]:.4f} rad")

    print("\n😎 观察现象：加入前馈 h (重力+科氏力) 后，误差以肉眼可见的速度收敛到 0，机械臂宛如在太空中失重一般！")
    
    # 保持画面不退出
    print("\n测试完成，关闭渲染窗口退出...")
    while sim.is_alive():
        sim.step()
        time.sleep(0.01)

if __name__ == "__main__":
    main()