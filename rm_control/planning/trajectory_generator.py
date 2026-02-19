import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    def __init__(self, freq=0.2, amp=0.25, duration=10.0, dt=0.001):
        """
        轨迹生成器基类
        :param freq: 运动频率 (Hz)
        :param amp: 运动幅度 (rad)
        :param duration: 轨迹总时长 (s)
        :param dt: 采样时间步长 (s)
        """
        self.freq = freq
        self.amp = amp
        self.duration = duration
        self.dt = dt
        
        # 🔥 标准 Ready Pose (伸展姿态)
        # J1-J7，确保腕关节(J6)在 1.8 左右，避免自碰撞
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.80, 0.785])
        
        # 预计算时间轴
        self.time_steps = np.arange(0, duration, dt)

    def get_state(self, t):
        """
        获取 t 时刻的目标状态
        :return: q_ref, dq_ref, ddq_ref (均为 np.array)
        """
        raise NotImplementedError("子类必须实现 get_state 方法")

    def plot_trajectory(self):
        """
        [调试工具] 直接画出轨迹曲线，用于检查合理性
        """
        qs, dqs, ddqs = [], [], []
        for t in self.time_steps:
            q, dq, ddq = self.get_state(t)
            qs.append(q)
            dqs.append(dq)
            ddqs.append(ddq)
        
        qs = np.array(qs)
        dqs = np.array(dqs)
        ddqs = np.array(ddqs)
        
        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 1. 位置
        for i in range(7):
            axes[0].plot(self.time_steps, qs[:, i], label=f'J{i+1}')
        axes[0].set_title("Joint Position (rad)")
        axes[0].set_ylabel("Pos")
        axes[0].legend(ncol=7, fontsize='x-small', loc='upper right')
        axes[0].grid(True)
        
        # 2. 速度
        for i in range(7):
            axes[1].plot(self.time_steps, dqs[:, i], label=f'J{i+1}')
        axes[1].set_title("Joint Velocity (rad/s)")
        axes[1].set_ylabel("Vel")
        axes[1].grid(True)

        # 3. 加速度
        for i in range(7):
            axes[2].plot(self.time_steps, ddqs[:, i], label=f'J{i+1}')
        axes[2].set_title("Joint Acceleration (rad/s^2)")
        axes[2].set_ylabel("Acc")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig("trajectory_check.png")
        print("✅ 轨迹检查图已保存至 trajectory_check.png")
        # plt.show() # 如果在服务器或无法弹窗的环境，请注释掉这行


class SineWaveTrajectory(TrajectoryGenerator):
    """
    关节空间正弦波轨迹 (用于测试电机响应)
    """
    def get_state(self, t):
        omega = 2 * np.pi * self.freq
        
        q_ref = np.zeros(7)
        dq_ref = np.zeros(7)
        ddq_ref = np.zeros(7)
        
        for i in range(7):
            # 相位差：让机械臂动起来像波浪，而不是整体点头
            phase = i * 0.5 
            
            # 幅度衰减：根部关节幅度小(0.5x)，腕部幅度大(1.0x)
            current_amp = self.amp * (0.5 if i < 2 else 1.0)
            
            # 计算理论公式
            # Pos: q0 + A * sin(wt + phi)
            q_ref[i] = self.q_home[i] + current_amp * np.sin(omega * t + phase)
            
            # Vel: A * w * cos(wt + phi)
            dq_ref[i] = current_amp * omega * np.cos(omega * t + phase)
            
            # Acc: -A * w^2 * sin(wt + phi)
            ddq_ref[i] = -current_amp * (omega**2) * np.sin(omega * t + phase)
            
        return q_ref, dq_ref, ddq_ref


class StepTrajectory:
    def __init__(self, target_joint_id, start_val, end_val, step_time=0.5):
        """
        Args:
            start_val: 起始角度 (t < step_time)
            end_val:   目标角度 (t >= step_time)
        """
        self.id = target_joint_id
        self.start_val = start_val # 新增：记录起点
        self.end_val = end_val     # 新增：记录终点
        self.t_step = step_time
        
        # 定义一个安全的初始姿态 (Panda Ready Pose)
        # J4 初始值得设为负数，防止一开始就撞墙
        self.q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        
        # 强制覆盖当前测试关节的初始值
        self.q_home[self.id] = self.start_val

    def get_state(self, t):
        q_ref = self.q_home.copy()
        
        # 阶跃逻辑
        if t >= self.t_step:
            q_ref[self.id] = self.end_val
        else:
            q_ref[self.id] = self.start_val
            
        return q_ref, np.zeros(7), np.zeros(7)



# ==============================================================================
# 单元测试 (Unit Test)
# 直接运行这个文件，可以检查轨迹是否正常
# ==============================================================================
if __name__ == "__main__":
    print("🧪 正在测试轨迹生成模块...")
    
    # 实例化一个正弦轨迹
    traj = SineWaveTrajectory(freq=0.5, amp=0.3, duration=5.0)
    
    # 打印 t=0 时的状态 (也就是机器人的起始状态)
    q0, dq0, ddq0 = traj.get_state(0)
    print(f"📍 起始姿态 (t=0):\n{q0}")
    
    # 检查是否有非法值 (NaN)
    if np.isnan(q0).any():
        print("❌ 错误：生成的轨迹包含 NaN！")
    else:
        print("✅ 数据完整性检查通过")
        
    # 画图检查
    traj.plot_trajectory()