import time
import numpy as np
import mujoco
import mujoco.viewer
import collections

# 🔥 [关键] 必须放在最前面！解决 macOS 下 MuJoCo 和 Matplotlib 的 GUI 冲突
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==============================================================================
# 1. 仿真接口 (集成安全力矩模式)
# ==============================================================================
class SimInterface:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 尝试寻找末端用于可视化 (Panda 通常是 link7 或 hand)
        self.ee_site_name = "panda_link7" 
        try:
            self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_site_name)
        except:
            print(f"⚠️ 未找到末端 {self.ee_site_name}, 使用默认末端 ID")
            self.ee_body_id = self.model.nbody - 1 

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        return self.data.qpos[:7].copy(), self.data.qvel[:7].copy()

    def set_torque(self, tau):
        self.data.ctrl[:7] = tau

    def set_control_mode(self, mode="torque"):
        # Panda 真实力矩极限
        max_torques = [87, 87, 87, 87, 12, 12, 12]
        for i in range(7):
            aid = i 
            if mode == "torque":
                limit = max_torques[i]
                # 1. 暴力放开输入限制
                self.model.actuator_ctrlrange[aid] = [-limit, limit]
                self.model.actuator_forcerange[aid] = [-limit, limit]
                
                # 2. 移除被动特性 (Affine)
                self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE
                self.model.actuator_biasprm[aid, :] = 0
                
                # 3. 纯增益
                self.model.actuator_gaintype[aid] = mujoco.mjtGain.mjGAIN_FIXED
                self.model.actuator_gainprm[aid, 0] = 1.0
                self.model.actuator_dyntype[aid] = mujoco.mjtDyn.mjDYN_NONE

        print(f"🛠️  模式切换至: {mode.upper()} (力矩封印已解除)")

    def get_ee_pos(self):
        return self.data.xpos[self.ee_body_id].copy()

    def calc_fk(self, q):
        """计算目标关节角的末端位置用于画图 (不改变物理状态)"""
        q_backup = self.data.qpos[:7].copy()
        self.data.qpos[:7] = q
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.data.xpos[self.ee_body_id].copy()
        
        # 恢复状态
        self.data.qpos[:7] = q_backup
        mujoco.mj_kinematics(self.model, self.data) 
        return pos

# ==============================================================================
# 2. 轨迹生成器 (已调整 Joint 6 中心为 1.8)
# ==============================================================================
class TrajectoryGenerator:
    def __init__(self):
        # 🔥 [核心修改] Ready Pose
        # Joint 4 (Elbow): -2.356 (手肘向后上方)
        # Joint 6 (Wrist): 1.800  (手腕向前伸展，避免碰撞)
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.80, 0.785])
        
        self.freq = 0.2  # 频率 0.2Hz
        self.amp = 0.25  # 幅度稍微调小一点，保证在 1.8 附近晃动不撞车

    def get_target(self, t):
        omega = 2 * np.pi * self.freq
        
        # 必须初始化为全0数组
        q_ref = np.zeros(7)
        dq_ref = np.zeros(7)
        ddq_ref = np.zeros(7)
        
        for i in range(7):
            phase = i * 0.5 
            
            # 让根部关节幅度小，中间关节幅度大
            current_amp = self.amp * (0.5 if i < 2 else 1.0)
            
            # 1. 位置
            q_ref[i] = self.q_home[i] + current_amp * np.sin(omega * t + phase)
            
            # 2. 速度
            dq_ref[i] = current_amp * omega * np.cos(omega * t + phase)
            
            # 3. 加速度
            ddq_ref[i] = -current_amp * (omega**2) * np.sin(omega * t + phase)
            
        return q_ref, dq_ref, ddq_ref

# ==============================================================================
# 3. 控制器 (PD + Gravity Compensation)
# ==============================================================================
class PDGravityController:
    def __init__(self, kp, kd):
        self.name = "PD + Gravity Comp"
        self.kp = np.diag(kp)
        self.kd = np.diag(kd)

    def compute_torque(self, q, dq, q_ref, dq_ref, ddq_ref, model, data):
        # 1. PD 控制
        e = q_ref - q
        de = dq_ref - dq
        tau_pd = self.kp @ e + self.kd @ de
        
        # 2. 重力补偿 (使用 MuJoCo 原生 Inverse Dynamics)
        original_qacc = data.qacc.copy()
        
        # 设加速度为0，计算维持当前状态所需的力 (重力+科氏力)
        data.qacc[:7] = 0 
        mujoco.mj_fwdPosition(model, data) 
        mujoco.mj_inverse(model, data)     
        tau_g = data.qfrc_inverse[:7].copy()
        
        # 恢复现场
        data.qacc[:] = original_qacc
        
        return tau_pd + tau_g

# ==============================================================================
# 4. Benchmark 运行器
# ==============================================================================
class BenchmarkRunner:
    def __init__(self, xml_path, controller):
        self.sim = SimInterface(xml_path)
        self.controller = controller
        
        # 数据记录
        self.history = {"t": [], "q": [], "dq": [], "tau": [], "q_ref": [], "dq_ref": []}
        
        # 可视化缓存
        self.trace_real = collections.deque(maxlen=100)
        self.trace_target = collections.deque(maxlen=100)

    def run(self, duration=8.0):
        print(f"🚀 开始测试: {self.controller.name}")
        self.sim.set_control_mode("torque")
        
        # 初始化轨迹生成器 (不需要参数)
        traj = TrajectoryGenerator()
        
        # 🔥 [关键步骤] 瞬移到起点
        # 获取 t=0 时的目标姿态，强制设置给机器人，防止开局飞掉
        q_start, _, _ = traj.get_target(0)
        self.sim.data.qpos[:7] = q_start
        self.sim.data.qvel[:7] = 0
        
        # 刷新一下运动学，确保 xpos 是新的
        mujoco.mj_fwdPosition(self.sim.model, self.sim.data)
        
        print("👀 正在启动 Viewer...")
        # 启动 Viewer
        with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
            print("✅ Viewer 启动成功！")
            start_time = time.time()
            sim_t = 0
            
            while viewer.is_running() and sim_t < duration:
                step_start = time.time()
                sim_t = self.sim.data.time

                # A. 获取数据
                q, dq = self.sim.get_state()
                q_ref, dq_ref, ddq_ref = traj.get_target(sim_t)
                
                # B. 计算控制
                tau = self.controller.compute_torque(q, dq, q_ref, dq_ref, ddq_ref, self.sim.model, self.sim.data)
                self.sim.set_torque(tau)
                self.sim.step()
                
                # C. 记录
                self.history["t"].append(sim_t)
                self.history["q"].append(q)
                self.history["dq"].append(dq)
                self.history["tau"].append(tau)
                self.history["q_ref"].append(q_ref)
                self.history["dq_ref"].append(dq_ref)
                
                # D. 可视化 (降频处理，防止卡顿)
                if int(sim_t * 1000) % 50 == 0:
                    pos_real = self.sim.get_ee_pos()
                    pos_target = self.sim.calc_fk(q_ref)
                    self.trace_real.append(pos_real)
                    self.trace_target.append(pos_target)
                    self._draw_scene(viewer, pos_target, pos_real)
                
                viewer.sync()
                
                # E. 控速
                time_until_next = self.sim.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

    def _draw_scene(self, viewer, target_pos, real_pos):
        # 防止几何体溢出
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom - 20:
            viewer.user_scn.ngeom = 0
        
        # 每次重置 geom 计数器，重新画
        viewer.user_scn.ngeom = 0 
        
        # 1. 目标球 (红, 半透明)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0, 0], 
            pos=target_pos, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.5]
        )
        viewer.user_scn.ngeom += 1

        # 2. 真实球 (绿, 实心)
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], 
            pos=real_pos, mat=np.eye(3).flatten(), rgba=[0, 1, 0, 1]
        )
        viewer.user_scn.ngeom += 1
        
        # 3. 轨迹点 (面包屑)
        for pos in self.trace_target:
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom: break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], 
                pos=pos, mat=np.eye(3).flatten(), rgba=[1, 0, 0, 0.2]
            )
            viewer.user_scn.ngeom += 1

    def plot_results(self):
        # 放到 Viewer 关闭后执行，避免 macOS GUI 冲突
        h = self.history
        t = np.array(h["t"])
        q = np.array(h["q"])      
        q_ref = np.array(h["q_ref"])
        dq = np.array(h["dq"])
        dq_ref = np.array(h["dq_ref"])
        tau = np.array(h["tau"])
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Benchmark: {self.controller.name}", fontsize=16)

        # 选最活跃的关节 J6 (Index 5)
        j_idx = 5 
        
        # Pos
        ax = axes[0, 0]
        ax.plot(t, q_ref[:, j_idx], 'r--', label="Target", lw=2)
        ax.plot(t, q[:, j_idx], 'b-', label="Real", lw=1.5)
        ax.set_title(f"Position (Joint {j_idx+1})")
        ax.legend()
        ax.grid(True)
        
        # Vel
        ax = axes[0, 1]
        ax.plot(t, dq_ref[:, j_idx], 'r--', label="Target Vel")
        ax.plot(t, dq[:, j_idx], 'b-', label="Real Vel")
        ax.set_title(f"Velocity (Joint {j_idx+1})")
        ax.legend()
        ax.grid(True)
        
        # Torque
        ax = axes[1, 0]
        ax.plot(t, tau[:, j_idx], 'g-')
        ax.set_title(f"Torque (Joint {j_idx+1})")
        ax.set_ylabel("Nm")
        ax.grid(True)
        
        # All Errors
        ax = axes[1, 1]
        error = (q_ref - q) * 180 / np.pi
        for i in range(7):
            ax.plot(t, error[:, i], label=f"J{i+1}")
        ax.set_title("Tracking Error (deg)")
        ax.legend(ncol=2, fontsize='small')
        ax.grid(True)
        
        plt.tight_layout()
        
        # 🔥🔥🔥 修改这里：不弹窗，直接保存！🔥🔥🔥
        save_path = "benchmark_result.png"
        print(f"💾 正在保存结果图表到: {save_path} ...")
        plt.savefig(save_path, dpi=300)
        print("✅ 保存成功！请在文件夹中查看图片。")
        
        # 可选：如果你非要看，可以用系统命令打开它
        # import os
        # os.system(f"open {save_path}")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    # ⚠️ 请修改为你的 XML 路径
    XML_PATH = "/Users/chenxu/Library/CloudStorage/OneDrive-Personal/Code/robot_dynamic_ctl_demo/rm_control/assets/franka_emika_panda/scene.xml"
    
    # 强力 PD 参数 (针对力矩控制优化)
    # J1-J4 (大关节): KP=800, KD=40
    # J5-J7 (小关节): KP=100-300, KD=10-20
    kp = np.array([800, 800, 800, 800, 300, 300, 100])
    kd = np.array([40,  40,  40,  40,  20,  20,  10])
    
    controller = PDGravityController(kp, kd)
    runner = BenchmarkRunner(XML_PATH, controller)
    
    runner.run(duration=8.0)
    
    # 必须等仿真窗口关闭后，才会画图
    runner.plot_results()