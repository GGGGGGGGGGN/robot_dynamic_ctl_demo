import numpy as np

class TrajectoryGenerator:
    def __init__(self, q_start: np.ndarray, q_end: np.ndarray, duration: float):
        """
        五次多项式轨迹生成器 (Quintic Polynomial)
        保证位置、速度、加速度连续且起始/结束速度加速度为0
        """
        self.q_start = np.array(q_start)
        self.q_end = np.array(q_end)
        self.duration = duration
        self.dq = self.q_end - self.q_start

    def get_state(self, t: float):
        """
        获取 t 时刻的状态
        Returns:
            q_d: 期望位置
            dq_d: 期望速度
            ddq_d: 期望加速度
        """
        if t < 0:
            return self.q_start, np.zeros_like(self.q_start), np.zeros_like(self.q_start)
        
        if t >= self.duration:
            return self.q_end, np.zeros_like(self.q_end), np.zeros_like(self.q_end)

        # 归一化时间 s = t / T
        s = t / self.duration
        
        # 五次多项式 s(t) = 10s^3 - 15s^4 + 6s^5
        s_pos = 10 * s**3 - 15 * s**4 + 6 * s**5
        s_vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.duration
        s_acc = (60 * s - 180 * s**2 + 120 * s**3) / (self.duration**2)
        
        q_d = self.q_start + self.dq * s_pos
        dq_d = self.dq * s_vel
        ddq_d = self.dq * s_acc
        
        return q_d, dq_d, ddq_d
