import numpy as np
import time

# 导入仿真与动力学基石
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_xml, get_model_path_urdf

# 🔥 导入你刚刚写的纯净版空间魔法控制器
from rm_control.controllers.controllers import CartesianImpedanceController

def main():
    print("🚀 启动三维操作空间 (OSC) 魔法测试...")
    
    # =====================================================================
    # 1. 实例化环境 (这次不挂 5kg 铁块了，让你纯粹体验弹簧的手感)
    # =====================================================================
    sim = SimInterface(
        xml_path=get_model_path_xml(), 
        render=True,     # 必须打开渲染，因为你要用鼠标和它互动！
        dt=0.001
    )
    sim.set_control_mode("torque")
    dyn_model = PinocchioDynamics(urdf_path=get_model_path_urdf())

    # =====================================================================
    # 2. 召唤笛卡尔阻抗控制器 (设定空间弹簧的软硬度)
    # =====================================================================
    # 空间刚度 K_x: XYZ 极其坚硬 (1000)，旋转姿态中等偏软 (50)
    K_x = [1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0]
    
    # 空间阻尼 D_x: 临界阻尼公式 D = 2 * sqrt(K)，防止弹簧来回震荡
    # 2*sqrt(1000) ≈ 63.2， 2*sqrt(50) ≈ 14.1
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
    # 然后以此为基准，我们在 Z 轴 (高度) 上强行往上提 15 厘米！
    dyn_model.update(q_start, np.zeros(7))
    pos_initial, rot_initial = dyn_model.compute_forward_kinematics(q_start)
    
    # 目标位置：X、Y 不变，Z 轴升高 0.15 米
    pos_ref = pos_initial.copy()
    pos_ref[2] += 0.15 
    
    # 目标姿态：保持初始姿态不变
    rot_ref = rot_initial.copy()

    print("\n" + "="*50)
    print(f"🎯 目标空间坐标 (XYZ): {np.round(pos_ref, 3)}")
    print("✨ 玩法说明：")
    print(" 1. 启动后，机械臂会瞬间向上拔高 15cm，定死在目标点。")
    print(" 2. 请在 MuJoCo 窗口中，双击机械臂的【末端手抓】，长按鼠标右键拖拽！")
    print(" 3. 接着，双击机械臂的【手肘】，长按鼠标右键拖拽（体验零空间魔法）！")
    print("="*50 + "\n")

    # =====================================================================
    # 4. 实时控制循环
    # =====================================================================
    duration = 30.0 # 给你 30 秒的尽情游玩时间
    steps = int(duration / sim.dt)
    
    print("🕹️ 交互游乐场已开放，尽情推拉吧！(30秒后自动关闭)...")
    
    for i in range(steps):
        if not sim.is_alive(): 
            break
            
        # 1. 获取真实状态
        q, dq = sim.get_state()
        
        # 2. 传入目标坐标，算出 7 个电机的空间抗衡力矩
        # 注意：这里要求静止保位，所以 vel_ref 默认是 0
        tau_cmd = ctrl_osc.update(q, dq, pos_ref, rot_ref)
        
        # 3. 下发力矩，步进仿真
        sim.set_joint_torque(tau_cmd)
        sim.step()
        
        # 每隔 2 秒打印一次当前位置，让你看看它多准
        if i % 2000 == 0:
            dyn_model.update(q, dq)
            pos_curr, _ = dyn_model.compute_forward_kinematics(q)
            err_norm = np.linalg.norm(pos_ref - pos_curr)
            print(f"[t={i/1000:.1f}s] 当前末端位置: {np.round(pos_curr, 3)} | 三维空间误差: {err_norm*1000:.1f} mm")

    print("\n测试完成，关闭渲染窗口...")
    if sim.viewer is not None:
        sim.viewer.close()

if __name__ == "__main__":
    main()