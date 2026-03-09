import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 基类定义
# ==============================================================================
class TrajectoryGenerator:
    def __init__(self, duration=10.0, dt=0.001):
        """
        轨迹生成器基类
        :param duration: 轨迹总时长 (s)
        :param dt: 采样时间步长 (s)
        """
        self.duration = duration
        self.dt = dt
        # 预计算时间轴，主要用于画图和预存
        self.time_steps = np.arange(0, duration, dt)
        
    def print_info(self):
        """标准化的轨迹信息打印"""
        print("\n" + "="*50)
        print(f"📍 轨迹配置信息: {self.__class__.__name__}")
        print(f"   - 总时长: {self.duration} s")
        print(f"   - 采样步长: {self.dt} s")
        if hasattr(self, 'q_target'):
            print(f"   - 目标姿态: {np.round(self.q_target, 3)}")
        if hasattr(self, 'freq'):
            print(f"   - 运动频率: {self.freq} Hz")
        print("="*50 + "\n")
        
    def get_state(self, t):
        """
        获取 t 时刻的目标状态
        :return: q_ref, dq_ref, ddq_ref (均为 np.array, shape=(7,))
        """
        raise NotImplementedError("子类必须实现 get_state 方法")

    def plot_trajectory(self, filename="trajectory_check.png"):
        """
        [调试工具] 自动遍历时间轴，画出位置、速度、加速度曲线
        """
        qs, dqs, ddqs = [], [], []
        for t in self.time_steps:
            q, dq, ddq = self.get_state(t)
            qs.append(q)
            dqs.append(dq)
            ddqs.append(ddq)
        
        qs, dqs, ddqs = np.array(qs), np.array(dqs), np.array(ddqs)
        
        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        titles = ["Joint Position (rad)", "Joint Velocity (rad/s)", "Joint Acceleration (rad/s^2)"]
        ylabels = ["Pos", "Vel", "Acc"]
        data_list = [qs, dqs, ddqs]
        
        for ax, data, title, ylabel in zip(axes, data_list, titles, ylabels):
            for i in range(7):
                ax.plot(self.time_steps, data[:, i], label=f'J{i+1}')
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            if ax == axes[0]:
                ax.legend(ncol=7, fontsize='x-small', loc='upper right')
                
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(filename)
        print(f"✅ 轨迹检查图已保存至 {filename}")


# ==============================================================================
# 具体轨迹实现类
# ==============================================================================
class FixedTrajectory(TrajectoryGenerator):
    """
    固定姿态轨迹：用于测试控制器在极端姿态下的静态保持能力 (对抗稳态误差和重力)
    """
    def __init__(self, q_target=None, duration=10.0, dt=0.001):
        super().__init__(duration, dt)
        
        if q_target is None:
            # 默认使用高重力力矩挑战姿态：关节 2 (肩) 和 关节 4 (肘) 承受极大扭矩
            self.q_target = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785], dtype=float)
        else:
            self.q_target = np.array(q_target, dtype=float)

    def get_state(self, t):
        """
        无论时间 t 是多少，始终返回固定的目标位置，以及为 0 的速度和加速度
        """
        q_ref = self.q_target.copy()
        dq_ref = np.zeros(7)
        ddq_ref = np.zeros(7)
        
        return q_ref, dq_ref, ddq_ref
    
class SineTrajectory(TrajectoryGenerator):
    """
    灵活的正弦轨迹：支持为每个关节独立设置初始位姿和振幅 (你刚刚实验用的版本)
    """
    def __init__(self, q_init, amplitude, freq=1.0, duration=10.0, dt=0.001):
        super().__init__(duration, dt)
        self.q_init = np.array(q_init, dtype=float)
        self.amplitude = np.array(amplitude, dtype=float)
        self.freq = freq
        self.w = 2 * np.pi * self.freq

    def get_state(self, t):
        # 🔥 修改为 (1 - cos) 轨迹，保证开局速度绝对为 0，防止观测器被瞬间冲爆！
        q_ref = self.q_init + self.amplitude * (1 - np.cos(self.w * t))
        dq_ref = self.amplitude * self.w * np.sin(self.w * t)
        ddq_ref = self.amplitude * (self.w**2) * np.cos(self.w * t)
        
        return q_ref, dq_ref, ddq_ref

class PhaseSineTrajectory(TrajectoryGenerator):
    """
    带有相位差的波浪正弦轨迹：用于测试全身协调性 (你早期的版本)
    """
    def __init__(self, freq=0.2, amp=0.25, duration=10.0, dt=0.001):
        super().__init__(duration, dt)
        self.freq = freq
        self.amp = amp
        self.q_home = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.80, 0.785])
        self.w = 2 * np.pi * self.freq

    def get_state(self, t):
        q_ref, dq_ref, ddq_ref = np.zeros(7), np.zeros(7), np.zeros(7)
        for i in range(7):
            phase = i * 0.5 
            current_amp = self.amp * (0.5 if i < 2 else 1.0)
            
            q_ref[i] = self.q_home[i] + current_amp * np.sin(self.w * t + phase)
            dq_ref[i] = current_amp * self.w * np.cos(self.w * t + phase)
            ddq_ref[i] = -current_amp * (self.w**2) * np.sin(self.w * t + phase)
            
        return q_ref, dq_ref, ddq_ref


class StepTrajectory(TrajectoryGenerator):
    """
    阶跃轨迹：用于测试控制器的瞬态响应 (超调、上升时间)
    """
    def __init__(self, target_joint_id, start_val, end_val, step_time=0.5, duration=10.0, dt=0.001):
        super().__init__(duration, dt)
        self.id = target_joint_id
        self.start_val = start_val
        self.end_val = end_val
        self.t_step = step_time
        
        self.q_home = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785], dtype=float)
        self.q_home[self.id] = self.start_val

    def get_state(self, t):
        q_ref = self.q_home.copy()
        if t >= self.t_step:
            q_ref[self.id] = self.end_val
        return q_ref, np.zeros(7), np.zeros(7)


class CartesianSineTrajectory(TrajectoryGenerator):
    """
    笛卡尔空间往复直线轨迹：用于测试笛卡尔阻抗控制器的空间轨迹跟踪能力
    在基准位置和姿态上，沿指定空间向量方向做正弦往复运动。
    """
    def __init__(self, pos_init, rot_init, amplitude, freq=0.5, duration=10.0, dt=0.001):
        super().__init__(duration, dt)
        self.pos_init = np.array(pos_init, dtype=float)
        self.rot_init = np.array(rot_init, dtype=float)
        self.amplitude = np.array(amplitude, dtype=float) # 决定直线方向和长度的3D向量
        self.freq = freq
        self.w = 2 * np.pi * self.freq

    def get_state(self, t):
        # 使用 (1 - cos) 保证初始速度为0，缓慢启动
        pos_ref = self.pos_init + self.amplitude * (1 - np.cos(self.w * t))
        
        # 组装 6D 前馈速度 (前3维是线速度，后3维角速度为0)
        vel_ref = np.zeros(6)
        vel_ref[:3] = self.amplitude * self.w * np.sin(self.w * t)
        
        return pos_ref, self.rot_init, vel_ref

    def plot_trajectory(self, filename="cartesian_trajectory_check.png"):
        print(f"✅ 笛卡尔轨迹暂不生成关节空间绘图，跳过...")


# ==============================================================================
# 单元测试 (直接运行此文件进行测试)
# ==============================================================================
if __name__ == "__main__":
    print("🧪 正在测试轨迹生成模块...")
    
    # 使用你刚才提供的具体参数进行测试
    q_init = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785]
    amplitude = [0.8, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0] # 只有关节 1 和 4 在动
    
    traj = SineTrajectory(q_init=q_init, amplitude=amplitude, freq=1.0, duration=4.0)
    
    q0, dq0, ddq0 = traj.get_state(0.0)
    print(f"📍 初始位置 (t=0):\n{q0}")
    print(f"📍 初始速度 (t=0):\n{dq0}")  # 注意：余弦波在 t=0 时速度不为 0！
    
    # 自动画图检查
    traj.plot_trajectory("test_sine_traj.png")