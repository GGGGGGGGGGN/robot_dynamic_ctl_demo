import numpy as np
import os
# 假设你的模块结构如下 (按你的 rrm 包规划)
# from rrm.simulation.sim_interface import SimInterface
# from rrm.controllers.pd_controller import JointPDController
# from rrm.planning.trajectory import SineWaveTrajectory
# from rrm.utils.plotter import BenchmarkPlotter

# 为了演示，我这里直接引用之前的类 (假设它们都在当前环境可访问)
from rm_control.simulation.sim_interface import SimInterface
from rm_control.controllers.controllers import JointPDController,ComputedTorqueController
from rm_control.planning.trajectory_generator import SineWaveTrajectory
from rm_control.utils.plotter import BenchmarkPlotter
from rm_control.assets import get_model_path_xml, get_model_path_urdf
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.results import get_result_fig_dir

# 导入刚才写的 Runner
from rm_control.utils.benchmark import BenchmarkRunner

def main():
    # =========================================================
    # 1. 组装零件 (实例化所有模块)
    # =========================================================
    
    # A. 仿真环境 (SimInterface)
    xml_path = get_model_path_xml()
    result_fig_dir = get_result_fig_dir()

    # ⚠️ render=False: 因为 Runner 会负责启动 Viewer，这里不要重复启动
    sim = SimInterface(xml_path, dt=0.001, render=False)
    
    # B. 控制器 (Controller)
    # kp = np.array([300, 300, 300, 100, 100, 100, 20])
    # kd = np.array([40,  40,  40,  40,  20,  20,  10])
    kp_val = 100.0
    kd_val = 20.0
    
    # 全关节统一！CTC 的数学美感就在于此
    kp = np.array([kp_val] * 7)
    kd = np.array([kd_val] * 7)
   
    # ctrl_with_grav = JointPDController(kp, kd, pin_dyn=None)
    # ctc_ctrl = ComputedTorqueController(kp, kd, pin_dyn=pin_dyn)
    
    
    # C. 轨迹规划 (Trajectory)
    trajectory = SineWaveTrajectory(freq =2, amp = 0.4, duration=5.0, dt=0.001)

    # D. 动力学模型 (Pinocchio)
    urdf_path = get_model_path_urdf()
    pin_dyn = PinocchioDynamics(urdf_path, ee_name="panda_link7")

    # 把 sim 和 controller 传进去
    # ctrl_with_grav = JointPDController(kp, kd, pin_dyn=None)
    ctc_ctrl = ComputedTorqueController(kp, kd, pin_dyn=pin_dyn)
    runner = BenchmarkRunner(sim, ctc_ctrl)
    # runner = BenchmarkRunner(sim_interface=sim, controller=controller, pinocchio_interface=pin_dyn)
    
    # 开始跑
    history_data = runner.run(trajectory, duration=5.0)

    # =========================================================
    # 3. 后处理画图
    # =========================================================
    result_fig_name = 'crontrol_test_'+ctc_ctrl.name+'.png'
    result_fig_path = os.path.join(result_fig_dir, result_fig_name)
    plotter = BenchmarkPlotter(run_name="final_test2")
    plotter.plot(history_data, save_path=result_fig_path)

if __name__ == "__main__":
    main()