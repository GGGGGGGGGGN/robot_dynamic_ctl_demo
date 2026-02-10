import time
import numpy as np
from rm_control.simulation.sim_interface import SimInterface

def test_position_control():
    # 1. 初始化环境：开启 GUI，使用位置模式
    sim = SimInterface(mode='position', render=True)
    
    # 获取初始状态
    qpos, _ = sim.get_state()
    num_actuators = sim.nu
    
    print("🚀 开始位置控制测试... (机器人应该会挥手)")
    
    start_time = time.time()
    
    # 仿真主循环
    while sim.is_alive():
        t = sim.get_time()
        
        # === 简单的控制策略 ===
        # 让第 4 个关节 (通常是左臂肘部) 做正弦运动
        # 目标位置 = 初始位置 + 幅度 * sin(频率 * 时间)
        action = np.zeros(num_actuators)
        
        # 注意：这里假设 act_l3 是控制肘部的，具体看你的执行器列表顺序
        # 可以 print(sim.actuator_names) 确认索引
        joint_idx = 4 
        action[joint_idx] = 1.0 * np.sin(2.0 * t) 
        
        # 升降台保持不动 (或慢慢升起)
        action[0] = 0.5  # 升降台升到 0.5m
        
        # ====================
        
        # 执行一步
        sim.step(action)
        
        # 控制帧率 (可选，为了让人眼看清楚，否则 Python 跑太快了)
        time.sleep(sim.dt)

        # 运行 10 秒后自动退出
        if t > 10.0:
            print("✅ 测试结束")
            break
            
    sim.close()

if __name__ == "__main__":
    test_position_control()