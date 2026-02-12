import numpy as np

class JointPDController:
    def __init__(self, kp, kd, q_des=None):
        """
        关节空间 PD 控制器 + 动力学补偿
        Args:
            kp (np.array): 比例增益 (Stiffness)
            kd (np.array): 微分增益 (Damping)
            q_des (np.array): 初始目标位置 (可后续通过 set_target 更新)
        """
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.q_des = np.array(q_des) if q_des is not None else None
        self.dq_des = None

    def set_target(self, q_des, dq_des=None):
        """更新目标状态"""
        self.q_des = np.array(q_des)
        if dq_des is None:
            self.dq_des = np.zeros_like(q_des)
        else:
            self.dq_des = np.array(dq_des)

    def compute(self, q, dq, h):
        """
        计算控制力矩
        Args:
            q: 当前关节位置
            dq: 当前关节速度
            h: 动力学非线性项 (重力 + 科氏力)，来自 Pinocchio
        Returns:
            tau: 目标力矩
        """
        if self.q_des is None:
            return np.zeros_like(q)

        # 1. 计算误差
        e = self.q_des - q
        
        # 目标速度通常为 0，除非做轨迹跟踪
        target_v = self.dq_des if self.dq_des is not None else np.zeros_like(dq)
        de = target_v - dq

        # 2. PD 控制律 (Feedback)
        # 对应元素相乘
        tau_pd = self.kp * e + self.kd * de

        # 3. 动力学补偿 (Feedforward)
        # 加上 h 项，抵消重力，让机器人“感觉”自己没有重量
        tau = tau_pd + h
        
        return tau