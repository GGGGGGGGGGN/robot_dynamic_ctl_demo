import numpy as np
import mujoco
from rm_control.simulation.sim_interface import SimInterface
from rm_control.assets import get_model_path_xml

def verify_inverse_logic():
    print("⚖️  启动 MuJoCo 逆动力学逻辑验证...")
    
    # 1. 初始化仿真
    sim = SimInterface(get_model_path_xml(), render=False)
    
    # 2. 强制设置电机为纯力矩模式 (Gain=1, Bias=0)
    # 这一步是为了让 ctrl=10 直接产生 10Nm 的力，方便观察
    sim.set_control_mode("torque")
    sim.model.actuator_gainprm[:, 0] = 1.0
    sim.model.actuator_biasprm[:, :] = 0
    sim.model.dof_damping[:] = 0 # 为了数据干净，去掉阻尼

    # 3. 定义一个固定的物理状态 (Control Variate)
    # 我们随便定一个姿态，要求机器人保持这个加速度
    q_fix   = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7])
    dq_fix  = np.zeros(7)
    ddq_fix = np.zeros(7) # 设为0，即保持悬停所需的力（纯重力）

    print(f"🎯 目标状态: 保持姿态悬停 (ddq=0)")
    
    # ==========================================
    # 🧪 情况 A：我不出力 (ctrl = 0)
    # ==========================================
    print("\n[情况 A] 电机不出力 (Ctrl = 0)")
    
    # 填入状态
    sim.data.qpos[:7] = q_fix
    sim.data.qvel[:7] = dq_fix
    sim.data.qacc[:7] = ddq_fix
    
    # 设定指令为 0
    sim.data.ctrl[:7] = 0.0

    # 计算三部曲
    mujoco.mj_fwdPosition(sim.model, sim.data)  # 1. 算几何
    mujoco.mj_fwdActuation(sim.model, sim.data) # 2. 算电机力
    mujoco.mj_inverse(sim.model, sim.data)      # 3. 算补力

    # 记录数据
    act_A = sim.data.qfrc_actuator[:7].copy()
    inv_A = sim.data.qfrc_inverse[:7].copy()
    total_A = act_A + inv_A

    print(f"  -> 电机出力 (Actuator): {act_A[1]:.4f} Nm (关节2)")
    print(f"  -> 系统补力 (Inverse):  {inv_A[1]:.4f} Nm (关节2)")
    print(f"  -> 物理总需 (Total):    {total_A[1]:.4f} Nm")

    # ==========================================
    # 🧪 情况 B：我帮忙出点力 (ctrl = 20)
    # ==========================================
    print("\n[情况 B] 电机帮忙出 20Nm (Ctrl = 20)")
    
    # 再次填入完全相同的状态 (防止被修改)
    sim.data.qpos[:7] = q_fix
    sim.data.qvel[:7] = dq_fix
    sim.data.qacc[:7] = ddq_fix
    
    # 设定指令为 20
    sim.data.ctrl[:7] = 20.0 

    # 计算三部曲
    mujoco.mj_fwdPosition(sim.model, sim.data)
    mujoco.mj_fwdActuation(sim.model, sim.data)
    mujoco.mj_inverse(sim.model, sim.data)

    # 记录数据
    act_B = sim.data.qfrc_actuator[:7].copy()
    inv_B = sim.data.qfrc_inverse[:7].copy()
    total_B = act_B + inv_B

    print(f"  -> 电机出力 (Actuator): {act_B[1]:.4f} Nm (关节2)")
    print(f"  -> 系统补力 (Inverse):  {inv_B[1]:.4f} Nm (关节2)")
    print(f"  -> 物理总需 (Total):    {total_B[1]:.4f} Nm")

    # ==========================================
    # 📊 最终对比
    # ==========================================
    print("\n" + "="*40)
    print("💡 结论验证")
    print("="*40)
    
    diff_total = np.max(np.abs(total_A - total_B))
    diff_inv   = inv_A[1] - inv_B[1]
    
    print(f"1. 物理总需求力变化了没？ {diff_total:.6f} Nm (预期: 0.0)")
    
    if diff_total < 1e-5:
        print("   ✅ 物理一致性验证通过！无论你怎么给油门，总需求力矩是不变的。")
    
    print(f"2. Inverse 变化了多少？   {diff_inv:.4f} Nm")
    print(f"   (预期: 刚好等于我们增加的电机力 20.0 Nm)")

    if abs(diff_inv - 20.0) < 0.1:
        print("   ✅ 逻辑验证成功！Inverse 确实就是 '差价'。")
        print("      公式证明: Inv_New = Inv_Old - Delta_Ctrl")
    else:
        print("   ❌ 验证失败，数据对不上。")

if __name__ == "__main__":
    verify_inverse_logic()