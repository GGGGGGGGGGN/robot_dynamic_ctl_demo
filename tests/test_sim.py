import time
import numpy as np
import mujoco
from rm_control.simulation.sim_interface import SimInterface
from rm_control.assets import get_model_path_torque

def main():
    # 1. 初始化 (位置控制模式)
    sim = SimInterface(get_model_path_torque())

    while sim.is_alive():
        t = sim.get_time()
        
        # 2. 生成控制信号
        # 左臂：像波浪一样动
        left_cmd = np.sin(t) * np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) 
        # 右臂：保持不动 (0位)
        right_cmd = np.zeros(6)
        # 升降台：慢慢升起
        platform_cmd = [0.1 * t if t < 5 else 0.5]
        
        # 3. 分别设置指令 (互不干扰)
        sim.set_left_arm_cmd(left_cmd)
        sim.set_right_arm_cmd(right_cmd)
        sim.set_platform_cmd(platform_cmd)
        
        # 4. 执行仿真
        sim.step()
        
        # 5. 获取数据打印
        if t % 1.0 < 0.01:
            l_pos = sim.get_left_arm_qpos()
            print(f"Time: {t:.2f} | Left Arm Pos: {l_pos[:2]}...")
            
        time.sleep(0.002)

if __name__ == "__main__":
    main()