import numpy as np
import time
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.mujoco_dynamics import DynamicsServer
from rm_control.controllers.ctc_controller import CTCController

def main():
    # 1. 必须使用 'torque' 模式！
    sim = SimInterface(mode='torque', render=True)
    
    # 2. 初始化动力学服务 (传入 sim 内部的 model/data)
    dyn_server = DynamicsServer(sim.model, sim.data)
    
    # 3. 配置 CTC 控制器
    # Kp 可以给得很大，因为 CTC 已经消除了非线性，剩下的就是简单的二阶线性系统
    kp = 100.0
    kd = 2.0 * np.sqrt(kp) # 临界阻尼公式: 2 * sqrt(k)
    
    # 注意：这里我们简单地给所有关节一样的参数，实际可以分关节调
    ctc = CTCController(dyn_server, 
                        kp=[kp] * sim.nv, 
                        kd=[kd] * sim.nv)
    
    print("🔥 CTC 控制器启动！机器人应该会瞬间锁定在目标位置...")
    
    target_q = np.zeros(sim.nv)
    # 设定一个目标姿态（比如左臂抬起）
    # 这里的索引需要根据你的实际 mapping 填，这里只是示例
    # 假设 idx_act_left 对应的是左臂
    sim.set_left_arm_cmd(np.zeros(len(sim.idx_act_left))) # 先占位
    
    # 找到左臂关节在 NV 中的索引 (为了设置 target_q)
    # 简单起见，我们假设前6个是底座轮子(不控制)，后面是手臂
    # ⚠️ 严谨做法是用 sim.idx_jnt_left
    left_arm_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_left]
    target_q[left_arm_indices] = np.array([0, -0.5, 1.5, 0, 0.5, 0]) 

    while sim.is_alive():
        # 1. 获取当前状态 (全量)
        q_now, dq_now = sim.get_state()
        
        # 2. 计算力矩 (全量)
        # 目标：让机器人去 target_q，速度为 0
        tau_full = ctc.compute(q_now, dq_now, target_q)
        
        # 3. 提取驱动关节的力矩
        # CTC 算出来的是所有自由度（包括轮子）的力矩，但轮子没有电机
        # 我们只提取我们关心的部分发给执行器
        
        # 提取左臂力矩
        tau_left = tau_full[left_arm_indices]
        sim.set_left_arm_cmd(tau_left)
        
        # 右臂保持 0位 (即 target_q 其他部分为0)
        # 如果你想让右臂也抗重力悬停，你需要把 tau_full 里的右臂部分也发过去
        right_arm_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_right]
        tau_right = tau_full[right_arm_indices]
        sim.set_right_arm_cmd(tau_right)
        
        # 升降台
        plat_idx = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_platform]
        sim.set_platform_cmd(tau_full[plat_idx])

        # 控制头部 
        head_idx = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_head]
        # 从 CTC 算出的全量力矩中提取头部力矩
        tau_head = tau_full[head_idx] 
        sim.set_head_cmd(tau_head)
        
        # 4. 发送指令
        sim.step()
        time.sleep(sim.dt)

if __name__ == "__main__":
    main()