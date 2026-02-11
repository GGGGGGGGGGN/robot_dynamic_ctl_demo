import numpy as np

class TrajectoryGenerator:
    def __init__(self):
        pass

    def min_jerk(self, q_start, q_end, time_dur, t_now):
        """
        五次多项式插值 (Minimum Jerk)
        Args:
            q_start: 起始位置
            q_end:   终止位置
            time_dur: 运动总耗时
            t_now:    当前时间 (从0开始)
        Returns:
            q, dq, ddq: 当前时刻的目标位置、速度、加速度
        """
        # 归一化时间 tau = t / T
        if t_now >= time_dur:
            return q_end, np.zeros_like(q_end), np.zeros_like(q_end)
        
        tau = t_now / time_dur
        
        # 五次多项式系数: 10*t^3 - 15*t^4 + 6*t^5
        # s 是位置缩放因子 (0 -> 1)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        
        # ds/dtau = 30t^2 - 60t^3 + 30t^4
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / time_dur
        
        # dds/dtau = 60t - 180t^2 + 120t^3
        dds = (60 * tau - 180 * tau**2 + 120 * tau**3) / (time_dur ** 2)
        
        q_diff = q_end - q_start
        
        q_t = q_start + s * q_diff
        dq_t = ds * q_diff
        ddq_t = dds * q_diff
        
        return q_t, dq_t, ddq_t

    def sine_wave(self, q_center, amplitude, freq, t_now):
        """简单正弦波测试"""
        w = 2 * np.pi * freq
        q_t = q_center + amplitude * np.sin(w * t_now)
        dq_t = amplitude * w * np.cos(w * t_now)
        ddq_t = -amplitude * (w**2) * np.sin(w * t_now)
        return q_t, dq_t, ddq_t