import numpy as np
import time
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mujoco # 引入这个是为了调用 id2name

# 引入你的库
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.controllers.joint_pd import JointPDController
from rm_control.assets import get_model_path_xml, get_model_path_urdf

def main():
    # ---------------------------------------------------------
    # 1. 环境初始化
    # ---------------------------------------------------------

    # 加载 MuJoCo
    xml_path = get_model_path_xml()
    # ⚠️ 确保 render=True，否则 viewer 不会弹窗，但如果只打印参数 False 也可以
    sim = SimInterface(xml_path, render=True) 
    
    # 设置为纯力矩模式 (Gain=1, Bias=0)
    sim.set_control_mode("torque") 
    
    # ---------------------------------------------------------
    # 🔥🔥🔥 [新增] 打印驱动器参数体检报告 🔥🔥🔥
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🔍 驱动器参数体检报告 (Actuator Inspection)")
    print("="*60)
    print(f"{'ID':<3} | {'Ctrl Range (输入范围)':<20} | {'Force Range (输出范围)':<20} | {'Gain'}")
    print("-" * 60)
    
    # 遍历所有驱动器
    for i in range(sim.model.nu):
        # 获取输入范围 (actuator_ctrlrange)
        ctrl_min = sim.model.actuator_ctrlrange[i][0]
        ctrl_max = sim.model.actuator_ctrlrange[i][1]
        
        # 获取输出力限制 (actuator_forcerange)
        force_min = sim.model.actuator_forcerange[i][0]
        force_max = sim.model.actuator_forcerange[i][1]
        
        # 获取增益 (gainprm)
        gain = sim.model.actuator_gainprm[i][0]
        
        print(f"{i:<3} | [{ctrl_min:>7.2f}, {ctrl_max:>7.2f}]   | [{force_min:>7.2f}, {force_max:>7.2f}]   | {gain:>5.2f}")
    
    print("="*60 + "\n")
    # ---------------------------------------------------------

    # 加载 Pinocchio
    urdf_path = get_model_path_urdf()
    pin_dyn = PinocchioDynamics(urdf_path, ee_name="panda_link7")

    print("✅ 环境初始化完成，物理参数已清理。")

    while sim.viewer.is_running():
        step_start = time.time()
        sim.step()
        
        time_until_next_step = sim.model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()