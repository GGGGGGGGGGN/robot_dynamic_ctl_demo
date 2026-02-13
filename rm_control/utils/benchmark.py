import time
import numpy as np
import mujoco
import mujoco.viewer

class BenchmarkRunner:
    def __init__(self, sim_interface, controller):
        self.sim = sim_interface
        self.controller = controller
        
        self.history = {
            "t": [], "q": [], "dq": [], "tau": [], 
            "q_ref": [], "dq_ref": []
        }

    def run(self, trajectory, duration=8.0, visualize=False):
        """
        Args:
            trajectory: 轨迹生成器
            duration: 仿真持续时间
            visualize: 是否开启可视化。设为 False 可获得极速仿真。
        """
        print(f"🚀 开始测试: {self.controller.name} (可视化: {visualize})")
        
        # 1. 场景重置
        q_start = trajectory.get_state(0)
        if isinstance(q_start, tuple): q_start = q_start[0]

        self.sim.data.qpos[:7] = q_start
        self.sim.data.qvel[:7] = 0
        self.sim.set_control_mode("torque")
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # 2. 准备仿真环境
        sim_t = 0
        
        # 定义核心循环逻辑，减少重复代码
        def run_loop(viewer=None):
            nonlocal sim_t
            while sim_t < duration:
                step_start = time.time()
                sim_t = self.sim.data.time

                # A. 传感器读数
                q, dq = self.sim.get_state()
                
                # B. 期望轨迹
                traj_out = trajectory.get_state(sim_t)
                if isinstance(traj_out, tuple):
                    q_ref, dq_ref, ddq_ref = traj_out
                else:
                    q_ref, dq_ref, ddq_ref = traj_out, np.zeros(7), np.zeros(7)
                
                # C. 计算控制力矩
                tau = self.controller.update(q, dq, q_ref, dq_ref, ddq_ref)
                
                # D. 执行力矩
                self.sim.set_joint_torque(tau)
                self.sim.step()
                
                # E. 数据记录
                self.history["t"].append(sim_t)
                self.history["q"].append(q.copy())
                self.history["dq"].append(dq.copy())
                self.history["tau"].append(tau.copy())
                self.history["q_ref"].append(q_ref.copy())
                self.history["dq_ref"].append(dq_ref.copy())

                # --- 差异化处理 ---
                if viewer is not None:
                    viewer.sync()
                    # 仅在可视化时进行严格控速，匹配真实物理时间
                    remain = self.sim.dt - (time.time() - step_start)
                    if remain > 0: 
                        time.sleep(remain)
                    if not viewer.is_running():
                        break

        # 3. 根据参数决定是否启动 Viewer
        if visualize:
            print("👀 启动可视化窗口...")
            with mujoco.viewer.launch_passive(self.sim.model, self.sim.data) as viewer:
                run_loop(viewer)
        else:
            print("⚡ 正在进行极速仿真 (Headless mode)...")
            start_wall_time = time.time()
            run_loop(None)
            end_wall_time = time.time()
            print(f"⏱️ 仿真完成！实际耗时: {end_wall_time - start_wall_time:.4f}s (仿真时长: {duration}s)")
        
        print("✅ 测试任务完成。")
        return self.history