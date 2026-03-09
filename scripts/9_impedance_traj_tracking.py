import numpy as np
import time

# 导入仿真与动力学基石
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_xml, get_model_path_urdf

# 🔥 导入你刚刚写的纯净版空间魔法控制器
from rm_control.controllers.controllers import CartesianImpedanceController

# 导入笛卡尔线段轨迹生成器
from rm_control.planning.trajectory_generator import CartesianSineTrajectory

def main():
    print("🚀 启动三维操作空间 (OSC) 魔法测试...")
    
    # =====================================================================
    # 1. 实例化环境 (这次不挂 5kg 铁块了，让你纯粹体验弹簧的手感)
    # =====================================================================
    sim = SimInterface(
        xml_path=get_model_path_xml(), 
        render=True,     # 关闭渲染，快速跑完画图
        dt=0.001
    )
    sim.set_control_mode("torque")
    dyn_model = PinocchioDynamics(urdf_path=get_model_path_urdf())

    # =====================================================================
    # 2. 召唤笛卡尔阻抗控制器 (设定空间弹簧的软硬度)
    # =====================================================================
    # 空间刚度 K_x: 暂时将 Z 轴刚度也调高至 1000，以确保轨迹跟踪的精度优先
    # 旋转姿态保持中等偏软(50)
    K_x = [1000.0,1000.0, 100.0, 50.0, 50.0, 50.0]
    
    # 空间阻尼 D_x: 临界阻尼公式 D = 2 * sqrt(K)，防止弹簧来回震荡
    # 2*sqrt(1000) ≈ 63.2, 2*sqrt(50) ≈ 14.1
    D_x = [63.2, 63.2, 63.2, 14.1, 14.1, 14.1] 
    
    # 零空间阻尼: 给多出来的第 7 个关节加点阻力，防止手肘乱甩
    null_damp = 10.0

    ctrl_osc = CartesianImpedanceController(
        pin_dyn=dyn_model, 
        K_x=K_x, 
        D_x=D_x, 
        null_damp=null_damp
    )

    # =====================================================================
    # 3. 设定“空间锚点” (Target Pose)
    # =====================================================================
    # 先把机器人摆到一个舒服的初始姿态
    q_start = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    sim.set_robot_state(q=q_start, dq=np.zeros(7))
    
    # 极其聪明的做法：直接用动力学库算一下这个姿态下末端在哪，
    # 直接以此为纯基准，防止产生巨大的初始阶跃误差跳变！
    dyn_model.update(q_start, np.zeros(7))
    pos_initial, rot_initial = dyn_model.compute_forward_kinematics(q_start)
    
    pos_base = pos_initial.copy()
    rot_base = rot_initial.copy()

    # 创建一条在水平面上来回折返的直线轨迹 (沿 Y 轴走 0.2 米)
    # 振幅向量 amplitude_cart 决定了直线的方向和长度
    amplitude_cart = [0.0, 0.1, 0.0]  
    traj = CartesianSineTrajectory(
        pos_init=pos_base, 
        rot_init=rot_base, 
        amplitude=amplitude_cart, 
        freq=1, 
        duration=30.0,
        dt=sim.dt
    )

    print("\n" + "="*50)
    print(f"🎯 预备空间坐标 (XYZ): {np.round(pos_base, 3)}")
    print("✨ 玩法说明 (轨迹+阻抗)：")
    print(" 1. 启动后，机械臂将在水平面内按照设定的直线来回运动！")
    print(" 2. 请在运动中，用鼠标拖拽机械臂末端：")
    print("    - 水平方向极其坚硬 (推拉阻力很大，始终恪守轨迹)。")
    print("    - Z轴方向柔如弹簧 (可以轻松将末端按下去或提起来)。")
    print("="*50 + "\n")

    # =====================================================================
    # 4. 实时控制循环
    # =====================================================================
    duration = 10.0 # 跑10秒画图
    steps = int(duration / sim.dt)
    
    print(f"� 开始记录轨迹数据，时长 {duration} 秒...")
    
    history_pos = []
    history_ref = []
    history_rot = []
    history_rot_ref = []
    
    for i in range(steps):
        if not sim.is_alive(): 
            break
            
        # 1. 获取真实状态
        q, dq = sim.get_state()
        
        # 2. 获取当前时刻的参考轨迹 (位置、姿态，以及前馈速度)
        t = i * sim.dt
        pos_ref, rot_ref, vel_ref = traj.get_state(t)
        
        # 3. 传入目标坐标和速度，算出 7 个电机的空间抗衡力矩
        tau_cmd = ctrl_osc.update(q, dq, pos_ref, rot_ref, vel_ref=vel_ref)
        
        # 4. 下发力矩，步进仿真
        sim.set_joint_torque(tau_cmd)
        sim.step()
        
        # 记录数据
        dyn_model.update(q, dq)
        pos_curr, rot_curr = dyn_model.compute_forward_kinematics(q)
        history_pos.append(pos_curr)
        history_ref.append(pos_ref)
        history_rot.append(rot_curr)
        history_rot_ref.append(rot_ref)
        
        if i % 2000 == 0:
            err_norm = np.linalg.norm(pos_ref - pos_curr)
            print(f"[t={i/1000:.1f}s] 目标位Z: {pos_ref[2]:.3f} | 实际Z: {pos_curr[2]:.3f} | 整体追踪误差: {err_norm*1000:.1f} mm")

    print("\n测试完成，开始画图...")
    if sim.viewer is not None:
        sim.viewer.close()

    history_pos = np.array(history_pos)
    history_ref = np.array(history_ref)
    history_rot = np.array(history_rot)
    history_rot_ref = np.array(history_rot_ref)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    
    time_ax = np.arange(len(history_pos)) * sim.dt
    
    # Convert rotation matrices to Euler angles (XYZ, in degrees)
    euler_curr = R.from_matrix(history_rot).as_euler('xyz', degrees=False)
    euler_ref = R.from_matrix(history_rot_ref).as_euler('xyz', degrees=False)
    
    # Unwrap phase to prevent 0 to 360 jumps, then convert to degrees
    for j in range(3):
        euler_curr[:, j] = np.unwrap(euler_curr[:, j])
        euler_ref[:, j] = np.unwrap(euler_ref[:, j])
        
    euler_curr = np.degrees(euler_curr)
    euler_ref = np.degrees(euler_ref)
    
    # Create a 2x2 layout to display XYZ in the same plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # --- 1. Position Tracking (XYZ on ONE plot) ---
    axes[0, 0].plot(time_ax, history_ref[:, 0], 'r--', label='Ref X')
    axes[0, 0].plot(time_ax, history_pos[:, 0], 'r-',  label='Actual X')
    axes[0, 0].plot(time_ax, history_ref[:, 1], 'g--', label='Ref Y')
    axes[0, 0].plot(time_ax, history_pos[:, 1], 'g-',  label='Actual Y')
    axes[0, 0].plot(time_ax, history_ref[:, 2], 'b--', label='Ref Z')
    axes[0, 0].plot(time_ax, history_pos[:, 2], 'b-',  label='Actual Z')
    axes[0, 0].set_title('XYZ Position Tracking')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # --- 2. Position Errors (XYZ on ONE plot) ---
    axes[0, 1].plot(time_ax, (history_ref[:, 0] - history_pos[:, 0]) * 1000, 'r-', label='X Error')
    axes[0, 1].plot(time_ax, (history_ref[:, 1] - history_pos[:, 1]) * 1000, 'g-', label='Y Error')
    axes[0, 1].plot(time_ax, (history_ref[:, 2] - history_pos[:, 2]) * 1000, 'b-', label='Z Error')
    axes[0, 1].set_title('XYZ Position Tracking Errors (mm)')
    axes[0, 1].set_ylabel('Error (mm)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # --- 3. Orientation Tracking (Roll, Pitch, Yaw on ONE plot) ---
    axes[1, 0].plot(time_ax, euler_ref[:, 0], 'r--', label='Ref Roll')
    axes[1, 0].plot(time_ax, euler_curr[:, 0], 'r-',  label='Actual Roll')
    axes[1, 0].plot(time_ax, euler_ref[:, 1], 'g--', label='Ref Pitch')
    axes[1, 0].plot(time_ax, euler_curr[:, 1], 'g-',  label='Actual Pitch')
    axes[1, 0].plot(time_ax, euler_ref[:, 2], 'b--', label='Ref Yaw')
    axes[1, 0].plot(time_ax, euler_curr[:, 2], 'b-',  label='Actual Yaw')
    axes[1, 0].set_title('Roll/Pitch/Yaw Orientation Tracking')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Angle (deg)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # --- 4. Orientation Errors (Roll, Pitch, Yaw on ONE plot) ---
    axes[1, 1].plot(time_ax, euler_ref[:, 0] - euler_curr[:, 0], 'r-', label='Roll Error')
    axes[1, 1].plot(time_ax, euler_ref[:, 1] - euler_curr[:, 1], 'g-', label='Pitch Error')
    axes[1, 1].plot(time_ax, euler_ref[:, 2] - euler_curr[:, 2], 'b-', label='Yaw Error')
    axes[1, 1].set_title('Roll/Pitch/Yaw Orientation Tracking Errors (deg)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error (deg)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_tracking.png')
    print("✅ 轨迹跟踪图已保存至 trajectory_tracking.png")

if __name__ == "__main__":
    main()
