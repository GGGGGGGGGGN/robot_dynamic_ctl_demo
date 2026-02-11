import numpy as np
import time
import os
import sys

# 这一行是为了确保能导入 rm_control 包，如果报错找不到包，请取消注释并修改路径
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.mujoco_dynamics import MujocoDynamics
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.ctc_controller import CTCController
from rm_control.planning.trajectory import TrajectoryGenerator

def main():
    # 1. 初始化仿真 (必须是 Torque 模式)
    print("🚀 正在启动仿真环境...")
    sim = SimInterface(mode='torque', render=True)
    
    # 2. 初始化动力学后端
    dyn = PinocchioDynamics(sim.model, sim.data)
    
    # 3. 初始化控制器
    # CTC 将系统线性化为二阶系统，Kp 可以给大一点
    kp_val = 100.0
    kd_val = 2.0 * np.sqrt(kp_val) # 临界阻尼公式
    
    # 为所有关节设置相同的增益
    ctc = CTCController(
        dynamics_backend=dyn,
        kp=[kp_val] * sim.nv,
        kd=[kd_val] * sim.nv
    )
    
    # 4. 初始化轨迹生成器
    traj_gen = TrajectoryGenerator()
    
    # 5. 定义任务：让左臂抬起
    # 获取左臂关节的全局索引
    left_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_left]
    
    # 定义起始点 (当前位置，全0)
    q_home = np.zeros(sim.nv)
    
    # 定义目标点 (Target)
    q_target = np.zeros(sim.nv)
    # 设定左臂目标姿态：抬肩、弯肘
    q_target[left_indices] = np.array([0, -0.6, 1.5, 0.5, 1.0, 0]) 
    
    # 运动参数
    duration = 2.0 # 2秒完成动作
    start_time = sim.get_time()
    
    print("✨ CTC 控制器已就绪，开始执行轨迹...")

    while sim.is_alive():
        # --- A. 获取时间与状态 ---
        t_curr = sim.get_time()
        t_rel = t_curr - start_time
        q_now, dq_now = sim.get_state()
        
        # --- B. 轨迹规划 (五次多项式) ---
        # 实时计算当前时刻应该在哪里
        q_des, dq_des, ddq_des = traj_gen.min_jerk(q_home, q_target, duration, t_rel)
        
        # --- C. 计算力矩 (CTC核心) ---
        # 这是一个全量力矩，包含了底盘、头部、双臂所有关节
        tau_full = ctc.compute(q_now, dq_now, q_des, dq_des, ddq_des)
        
        # --- D. 分发力矩 ---
        # 我们算出了全身的力矩，现在要分发给各个执行器接口
        
        # 1. 左臂：执行轨迹
        sim.set_left_arm_cmd(tau_full[left_indices])
        
        # 2. 右臂：虽然目标是0，但CTC算出的tau包含了重力补偿，所以它会悬停而不会掉下来
        right_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_left] # 修正：这里应该是 idx_jnt_right
        # 注意：上面的代码有一处笔误，应该是 sim.idx_jnt_right
        # 修正如下：
        right_real_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_right]
        sim.set_right_arm_cmd(tau_full[right_real_indices])
        
        # 3. 头部：抗重力
        head_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_head]
        sim.set_head_cmd(tau_full[head_indices])
        
        # 4. 升降台：保持位置
        plat_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_platform]
        sim.set_platform_cmd(tau_full[plat_indices])
        
        # --- E. 物理步进 ---
        sim.step()
        
        # 稍微加点延时，防止仿真跑太快看不清 (可选)
        time.sleep(sim.dt)

if __name__ == "__main__":
    main()