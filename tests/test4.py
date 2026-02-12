import numpy as np
import time
import os

# 1. 设置 Matplotlib 后端 (必须在 import pyplot 之前)
import matplotlib
matplotlib.use('Agg') # 使用非交互式后端，防止 macOS GUI 冲突
import matplotlib.pyplot as plt

# 2. 引入 Mujoco
import mujoco
import mujoco.viewer

# 3. 引入你的库 (假设这些文件都在正确的位置)
from rm_control.simulation.sim_interface import SimInterface
from rm_control.assets import get_model_path_xml

# ⚠️ 如果你还没有把 TrajectoryGenerator 封装进库，
# 请确保 trajectory_generator.py 在同一目录下，或者在这里直接定义它
try:
    from rm_control.utils.trajectory_generator import TrajectoryGenerator
except ImportError:
    # 为了方便你直接运行，如果找不到库，我就从本地 import
    # (假设你把刚才写的生成器保存为了 trajectory_generator.py)
    from trajectory_generator import TrajectoryGenerator

# ==============================================================================
# 定义一个简单的控制器 (为了保证能跑通，我们先在本地定义这个逻辑)
# ==============================================================================
class PDGravityController:
    def __init__(self, kp, kd):
        self.kp = np.diag(kp)
        self.kd = np.diag(kd)

    def compute_torque(self, q, dq, q_ref, dq_ref, ddq_ref, model, data):
        """
        计算控制力矩: PD + 重力补偿 (使用 MuJoCo ID)
        """
        # 1. PD 项
        e = q_ref - q
        de = dq_ref - dq
        tau_pd = self.kp @ e + self.kd @ de
        
        # 2. 重力补偿 (G + Coriolis)
        # 技巧: 设 qacc=0, mj_inverse 算出来的就是维持当前状态所需的力
        original_qacc = data.qacc.copy()
        data.qacc[:7] = 0
        
        # 必须刷新几何，确保计算准确
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_inverse(model, data)
        tau_g = data.qfrc_inverse[:7].copy()
        
        # 恢复现场
        data.qacc[:] = original_qacc
        
        return tau_pd + tau_g

# ==============================================================================
# 主函数
# ==============================================================================
def main():
    print("🚀 启动仿真主程序...")
    
    # --------------------------------------------------------------------------
    # 1. 配置参数
    # --------------------------------------------------------------------------
    duration = 8.0  # 仿真时长
    dt = 0.001      # 步长
    
    # 强力 PD 参数 (针对 Panda 力矩控制)
    kp = np.array([800, 800, 800, 800, 300, 300, 100])
    kd = np.array([40,  40,  40,  40,  20,  20,  10])

    # --------------------------------------------------------------------------
    # 2. 初始化模块
    # --------------------------------------------------------------------------
    xml_path = get_model_path_xml()
    sim = SimInterface(xml_path, dt=dt)
    
    # 初始化轨迹生成器 (内部已包含 Ready Pose)
    traj_gen = TrajectoryGenerator(duration=duration, dt=dt)
    
    # 初始化控制器
    controller = PDGravityController(kp, kd)

    # --------------------------------------------------------------------------
    # 3. 🔥 关键步骤: 预计算并渲染轨迹
    # --------------------------------------------------------------------------
    # 这会调用 SimInterface 里的 FK，把整条红线算出来存进缓存
    sim.precompute_trajectory(traj_gen)
    
    # --------------------------------------------------------------------------
    # 4. 准备仿真环境
    # --------------------------------------------------------------------------
    # 切换到纯力矩模式 (解除 1.76Nm 封印)
    sim.set_control_mode("torque")
    
    # 🌟 瞬移到起点: 防止机器人从 0000 姿态猛地弹开
    q_start, _, _ = traj_gen.get_state(0)
    sim.data.qpos[:7] = q_start
    sim.data.qvel[:7] = 0
    # 刷新一下，确保一开始就在正确位置
    mujoco.mj_fwdPosition(sim.model, sim.data)

    # 数据记录
    history = {"t": [], "q": [], "dq": [], "tau": [], "q_ref": []}

    # --------------------------------------------------------------------------
    # 5. 进入 MuJoCo 循环
    # --------------------------------------------------------------------------
    print("✨ 打开 Viewer，开始运动...")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        start_time = time.time()
        sim_t = 0
        
        while viewer.is_running() and sim_t < duration:
            loop_start = time.time()
            sim_t = sim.data.time
            
            # A. 获取状态
            q, dq = sim.get_state()
            
            # B. 获取轨迹目标
            q_ref, dq_ref, ddq_ref = traj_gen.get_state(sim_t)
            
            # C. 计算控制律
            tau = controller.compute_torque(q, dq, q_ref, dq_ref, ddq_ref, sim.model, sim.data)
            
            # D. 执行
            sim.set_torque(tau)
            sim.step()
            
            # E. 🔥 渲染轨迹线 (画出之前预计算的红点)
            # 这一步会把红色的参考路径画在屏幕上
            sim.draw_trajectory(viewer)
            
            viewer.sync()
            
            # F. 记录数据
            history["t"].append(sim_t)
            history["q"].append(q)
            history["dq"].append(dq)
            history["tau"].append(tau)
            history["q_ref"].append(q_ref)
            
            # G. 控速 (Real-time sync)
            time_until_next = sim.model.opt.timestep - (time.time() - loop_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    print("✅ 仿真结束，正在生成图表...")

    # --------------------------------------------------------------------------
    # 6. 后处理与绘图 (Agg 后端保存图片)
    # --------------------------------------------------------------------------
    plot_results(history)

def plot_results(history):
    """画图并保存，不弹窗"""
    t = np.array(history["t"])
    q = np.array(history["q"])
    q_ref = np.array(history["q_ref"])
    tau = np.array(history["tau"])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Simulation Results (Agg Backend)", fontsize=16)

    # 这里的索引 5 对应关节 6 (Wrist)，之前我们调过它的轨迹
    j_idx = 5 
    
    # 位置
    axes[0, 0].plot(t, q_ref[:, j_idx], 'r--', label="Ref", lw=2)
    axes[0, 0].plot(t, q[:, j_idx], 'b-', label="Real")
    axes[0, 0].set_title(f"Joint {j_idx+1} Position")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 力矩
    axes[1, 0].plot(t, tau[:, j_idx], 'g-')
    axes[1, 0].set_title(f"Joint {j_idx+1} Torque (Nm)")
    axes[1, 0].grid(True)
    
    # 误差
    error = (q_ref - q) * 180 / np.pi
    for i in range(7):
        axes[1, 1].plot(t, error[:, i], label=f"J{i+1}")
    axes[1, 1].set_title("Tracking Error (deg)")
    axes[1, 1].legend(ncol=2, fontsize='small')
    axes[1, 1].grid(True)

    # 保存
    save_path = "simulation_report.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"💾 结果已保存至: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    main()