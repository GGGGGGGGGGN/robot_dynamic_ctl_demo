import time
import numpy as np
import mujoco
import sys
import os

sys.path.append(os.getcwd())

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_torque, get_model_path_urdf

def main():
    print("🚀 开始 Pinocchio 集成测试 (自动裁剪版)...")

    # 1. 启动 MuJoCo 仿真
    xml_path = get_model_path_torque()
    sim = SimInterface(xml_path, render=True) # 你的 SimInterface
    
    # === 🔥 关键步骤：获取 MuJoCo 关节白名单 ===
    mj_joint_names = []
    for i in range(sim.model.njnt):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        mj_joint_names.append(name)
    
    print(f"📋 MuJoCo 关节 ({len(mj_joint_names)}个): {mj_joint_names}")

    # 2. 初始化 Pinocchio (传入白名单！)
    urdf_path = get_model_path_urdf()
    
    try:
        # 这里把 names 传进去，Pinocchio 就会自动把 wheel 锁死
        pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=mj_joint_names)
    except Exception as e:
        print(f"❌ Pinocchio 加载失败: {e}")
        return

    # 3. 验证维度是否对齐
    print(f"\n🔍 --- 维度检查 ---")
    print(f"MuJoCo nv: {sim.nv}")
    print(f"Pinocchio nv: {pin_dyn.nv}")
    
    if sim.nv != pin_dyn.nv:
        print("❌ 失败：维度依然不匹配！请检查名字是否完全一致。")
        return
    else:
        print("✅ 成功：维度完美对齐！")

    # 4. 循环测试
    print("\n🔄 开始动态循环...")
    start_time = time.time()
    
  # 定义目标位置 (比如全 0)
    q_target = np.zeros(sim.nv)
    
    # 定义 PD 参数 (刚度)
    kp = 50.0  # 弹簧硬度
    kd = 5.0   # 阻尼 (防止震荡)

    while sim.is_alive():
        q, dq = sim.get_state()
        
        # 1. 更新动力学模型
        pin_dyn.update(q, dq)
        M, h = pin_dyn.get_dynamics()
        
        # 2. 计算 PD 补偿项 (让它想回到 0)
        # tau_pd = Kp * (q_des - q) - Kd * dq
        tau_pd = kp * (q_target - q) - kd * dq
        
        # 3. 最终指令 = 重力补偿 + PD微调
        # h 负责托住重物，tau_pd 负责消除漂移
        tau_cmd = h + tau_pd 
        # 注意：严谨的 CTC 是 M @ (kp*e + kd*de) + h，
        # 但简单的 "重力补偿 + PD" (如上) 在定点控制时效果也很好，且更直观。
        
        sim.set_whole_body_cmd(tau_cmd)
        sim.step()
        
        time.sleep(sim.dt)

if __name__ == "__main__":
    main()