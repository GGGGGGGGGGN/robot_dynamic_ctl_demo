import numpy as np
import mujoco
import pinocchio as pin
import sys
import os

sys.path.append(os.getcwd())

from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_torque, get_model_path_urdf

def main():
    print("⚖️  开始严谨的模型一致性校验 (Fixed Version)...")
    
    # 1. 初始化
    xml_path = get_model_path_torque()
    sim = SimInterface(xml_path, render=False)
    
    mj_joints = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    urdf_path = get_model_path_urdf()
    pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=mj_joints)

    # 2. 获取参数
    limits_min = sim.model.jnt_range[:, 0]
    limits_max = sim.model.jnt_range[:, 1]
    dampings = sim.model.dof_damping

    np.random.seed(42)
    n_tests = 5
    
    print(f"\n🚀 开始 {n_tests} 次对比...")
    print(f"{'Item':<10} | {'Max Error':<12} | {'Relative':<10} | {'Status':<10}")
    print("-" * 55)

    for i in range(n_tests):
        # --- A. 生成随机状态 ---
        q_rand = np.random.uniform(limits_min * 0.95, limits_max * 0.95)
        dq_rand = np.random.uniform(-0.5, 0.5, sim.nv)
        ddq_rand = np.random.uniform(-0.2, 0.2, sim.nv)
        
        # --- B. MuJoCo 计算 (修正后) ---
        sim.data.qpos[:] = q_rand
        sim.data.qvel[:] = dq_rand
        sim.data.qacc[:] = ddq_rand
        

        mujoco.mj_forward(sim.model, sim.data)
        
        # 🔥 新增：碰撞检测 🔥
        # sim.data.ncon 是当前检测到的接触点数量
        if sim.data.ncon > 0:
            print(f"Test {i}: ⚠️ 跳过 (检测到 {sim.data.ncon} 个碰撞点，导致受力异常)")
            continue # 直接跳过本次循环
        
        # ⚠️ 修正点：只调用逆动力学！
        # mj_inverse 内部会自动处理必要的运动学更新
        sim.data.qacc[:] = ddq_rand
        
        mujoco.mj_inverse(sim.model, sim.data)
        tau_mj = sim.data.qfrc_inverse.copy()
        
        # --- C. Pinocchio 计算 ---
        pin_dyn.update(q_rand, dq_rand)
        tau_pin_rigid = pin.rnea(pin_dyn.model, pin_dyn.data, q_rand, dq_rand, ddq_rand)
        
        # --- D. 加上阻尼补偿 ---
        tau_pin_corrected = tau_pin_rigid + dampings * dq_rand
        
        # --- E. 误差分析 ---
        diff = np.abs(tau_mj - tau_pin_corrected)
        max_err_idx = np.argmax(diff)
        max_err = diff[max_err_idx]
        
        # 相对误差
        ref_val = np.abs(tau_mj[max_err_idx]) + 0.1
        rel_err = max_err / ref_val

        # 打印
        status = "✅ PASS" if max_err < 0.5 else "❌ FAIL"
        print(f"Test {i}: Max Err = {max_err:.4f} @ Joint {max_err_idx} | Rel: {rel_err:.2f} | {status}")
        
        if max_err > 0.5:
             print("   ⚠️  Mismatch Details (MJ vs Pin+Damp):")
             for j in range(sim.nv):
                 err = np.abs(tau_mj[j] - tau_pin_corrected[j])
                 if err > 0.5:
                     # 智能判定：是摩擦导致的吗？
                     # 如果误差接近 frictionloss (通常是 0.1~1.0)，那就是正常的
                     # MuJoCo摩擦力方向是 sign(dq)，我们可以尝试手动加一下看看
                     print(f"   J{j}: {tau_mj[j]:8.2f} vs {tau_pin_corrected[j]:8.2f} (Diff: {err:.2f})")
        print("-" * 55)

if __name__ == "__main__":
    main()