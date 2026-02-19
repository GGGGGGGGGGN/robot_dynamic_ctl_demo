import numpy as np
import os

from rm_control.simulation.sim_interface import SimInterface
from rm_control.controllers.controllers import *
from rm_control.planning.trajectory_generator import SineWaveTrajectory
from rm_control.utils.plotter import BenchmarkPlotter
from rm_control.assets import get_model_path_xml, get_model_path_urdf
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.results import get_result_fig_dir

from rm_control.utils.benchmark import BenchmarkRunner

xml_path = get_model_path_xml()
result_fig_dir = get_result_fig_dir()
sim = SimInterface(xml_path, dt=0.001, render=False)

urdf_path = get_model_path_urdf()
pin_dyn = PinocchioDynamics(urdf_path, ee_name="panda_link7")

# 2. 设置轨迹
trajectory = SineWaveTrajectory(freq=1.0, amp=0.1, duration=5.0, dt=0.001)

kp_val = 100.0
kd_val = 20.0

# 全关节统一！CTC 的数学美感就在于此
kp = np.array([kp_val] * 7)
kd = np.array([kd_val] * 7)

# 把 sim 和 controller 传进去
# ctrl_with_grav = JointPDController(kp, kd, pin_dyn=None)
sim.reset()
ctc_ctrl = ComputedTorqueController(kp, kd, pin_dyn=pin_dyn)
runner = BenchmarkRunner(sim, ctc_ctrl)

# 开始跑
history_data = runner.run(trajectory, duration=5.0, visualize=1)

# =========================================================
# 3. 后处理画图
# =========================================================
result_fig_name = 'crontrol_test_'+ctc_ctrl.name+".png"
result_fig_path = os.path.join(result_fig_dir, result_fig_name)
plotter = BenchmarkPlotter(run_name="final_test2")
plotter.plot(history_data, save_path=result_fig_path)

