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
    print("📐 开始雅可比矩阵 (Jacobian) 一致性校验 (Fixed Target)...")

    # 1. 初始化
    xml_path = get_model_path_torque()
    sim = SimInterface(xml_path, render=False)
    
    mj_joints = [mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    urdf_path = get_model_path_urdf()
    
    # 🔥🔥🔥 关键修改在这里 🔥🔥🔥
    # 我们强制指定要验证的末端名字，必须是 URDF 和 XML 里都有的！
    # 通常是 "r_link6" (右手) 或 "l_link6" (左手)
    TARGET_EE = "r_link6" 
    
    print(f"🎯 锁定测试目标: {TARGET_EE}")

    # 初始化 Pinocchio 时传入 ee_name
    try:
        pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=mj_joints, ee_name=TARGET_EE)
    except Exception as e:
        print(f"❌ Pinocchio 初始化失败: {e}")
        print("请检查 URDF 里有没有叫 'r_link6' 的 link。如果没有，请换成 'link6' 或 'hand' 试试。")
        return

    # 在 MuJoCo 里找到对应的 Body ID
    try:
        mj_body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_EE)
        if mj_body_id == -1:
            raise ValueError
        print(f"✅ MuJoCo Body Found: ID {mj_body_id}")
    except:
        print(f"❌ MuJoCo 找不到 Body: '{TARGET_EE}'")
        return

    # 2. 随机测试
    np.random.seed(42)
    n_tests = 5
    
    print(f"\n🚀 开始 {n_tests} 次对比...")
    print(f"{'Item':<10} | {'Max Lin Err':<12} | {'Max Ang Err':<12} | {'Status':<10}")
    print("-" * 60)

    for i in range(n_tests):
        # A. 随机姿态
        q_rand = np.random.uniform(-1.0, 1.0, sim.nv)
        dq_rand = np.zeros(sim.nv) 

        # B. MuJoCo 计算
        sim.data.qpos[:] = q_rand
        sim.data.qvel[:] = dq_rand
        mujoco.mj_forward(sim.model, sim.data)

        # MuJoCo 雅可比
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        target_point = sim.data.xpos[mj_body_id] # Body 原点
        mujoco.mj_jac(sim.model, sim.data, jacp, jacr, target_point, mj_body_id)
        J_mj = np.vstack([jacp, jacr])

        # C. Pinocchio 计算
        pin_dyn.update(q_rand, dq_rand)
        pin.computeJointJacobians(pin_dyn.model, pin_dyn.data, q_rand)
        
        # 使用 LOCAL_WORLD_ALIGNED (原点在末端，方向对齐世界)
        J_pin = pin.getFrameJacobian(
            pin_dyn.model, 
            pin_dyn.data, 
            pin_dyn.ee_id, 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )

        # D. 维度对齐
        if J_mj.shape != J_pin.shape:
            J_mj = J_mj[:, :pin_dyn.nv]

        # E. 误差分析
        diff = np.abs(J_mj - J_pin)
        err_lin = np.max(diff[:3, :])
        err_ang = np.max(diff[3:, :])

        # 稍微放宽一点点标准，因为浮点数计算方式不同
        status = "✅ PASS" if (err_lin < 1e-3 and err_ang < 1e-3) else "❌ FAIL"
        
        print(f"Test {i}: Lin Err={err_lin:.6f} | Ang Err={err_ang:.6f} | {status}")
        
        if "FAIL" in status:
             print("   ⚠️  Mismatch Details:")
             print("   可能原因：MuJoCo Body 的原点和 Pinocchio Link 的原点定义不重合。")
             print("   (例如：一个在法兰盘中心，一个在法兰盘表面)")

    print("-" * 60)

if __name__ == "__main__":
    main()