import numpy as np
import os
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_xml, get_model_path_urdf

def setup_robot_env():
    # 1. 获取模型路径
    xml_path = get_model_path_xml()
    urdf_path = get_model_path_urdf()
    
    # 2. 定义活跃关节名称 (必须与 URDF 中的名字一致)
    # 这 7 个关节在 XML 中现在也是唯一的 7 个转动关节 
    arm_joints = [
        "panda_joint1", "panda_joint2", "panda_joint3", 
        "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    
    # 3. 创建 MuJoCo 环境 (SimInterface)
    # 传入 arm_joints，它会自动处理 "panda_" 前缀的映射 
    sim = SimInterface(xml_path, active_joint_names=arm_joints, render=True)
    
    # 🔥 关键：将 MuJoCo 切换为纯力矩模式，指令单位变为 Nm 
    sim.set_control_mode("torque")
    
    # 4. 创建 Pinocchio 控制器环境 (PinocchioDynamics)
    # 指定末端执行器为 panda_link7 或 panda_hand 
    TARGET_EE = "panda_link7" 
    pin_dyn = PinocchioDynamics(urdf_path, active_joint_names=arm_joints, ee_name=TARGET_EE)
    
    # 5. 打印初始状态进行确认
    print("-" * 30)
    print(f"✅ MuJoCo 自由度 (nv): {sim.model.nv}")
    print(f"✅ Pinocchio 自由度 (nv): {pin_dyn.nv}")
    print(f"✅ 执行器数量 (nu): {sim.model.nu}")
    print("-" * 30)
    
    return sim, pin_dyn

if __name__ == "__main__":
    # 运行初始化
    sim, pin_dyn = setup_robot_env()