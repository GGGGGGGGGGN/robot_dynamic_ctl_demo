import time
import numpy as np
import mujoco
import mujoco.viewer

# 导入模块
from rm_control.simulation.sim_interface import SimInterface
from rm_control.planning.trajectory_generator import SineWaveTrajectory
from rm_control.assets import get_model_path_xml

def main():
    xml_path = get_model_path_xml()
    
    # ==========================================================================
    # 🔥 [修复 1] 必须设置 render=False
    # ==========================================================================
    # SimInterface 默认会开一个窗口，如果不关掉，后面 main 里又开一个，就会炸
    print("🤖 初始化 SimInterface...")
    sim = SimInterface(xml_path, dt=0.001, render=False) 

    print("📈 初始化 SineWaveTrajectory...")
    traj_gen = SineWaveTrajectory(duration=10.0, dt=0.001)

    # 预计算红线
    sim.precompute_trajectory(traj_gen)

    print("✨ 打开 Viewer，播放预设轨迹...")
    
    # 启动唯一的 Viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # 计算循环时间
            t = (time.time() - start_time) % traj_gen.duration
            
            # A. 获取目标角度
            q_ref, _, _ = traj_gen.get_state(t)
            
            # B. 修改机器人关节
            sim.data.qpos[:7] = q_ref
            
            # ==================================================================
            # 🔥 [修复 2] 使用 mj_kinematics 而不是 mj_forward
            # ==================================================================
            # mj_forward 会计算动力学(力、碰撞等)，这在“瞬移”时会导致不必要的计算和物理错误
            # mj_kinematics 只更新位置，速度极快，轨迹绝对对得上
            mujoco.mj_kinematics(sim.model, sim.data)
            
            # C. 画红线
            sim.draw_trajectory(viewer)
            
            # D. 同步画面
            viewer.sync()
            
            # ==================================================================
            # 🔥 [修复 3] 更平滑的帧率控制
            # ==================================================================
            # 简单的 sleep(0.01) 可能不准，尽量扣除计算时间
            time_until_next = 0.01 - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()