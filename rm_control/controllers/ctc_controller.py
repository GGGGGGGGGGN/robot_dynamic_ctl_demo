import numpy as np

class CTCController:
    def __init__(self, dynamics_backend, kp, kd):
        """
        Args:
            dynamics_backend: 可以是 MujocoDynamics 或 PinocchioDynamics
        """
        self.dyn = dynamics_backend
        self.kp = np.array(kp)
        self.kd = np.array(kd)

    def compute(self, q, dq, q_des, dq_des, ddq_des):
        # 1. 误差
        e = q_des - q
        de = dq_des - dq
        
        # 2. 从后端获取动力学
        # 这里体现了多态：控制器不关心是 MuJoCo 还是 Pinocchio 算的
        M, h = self.dyn.get_dynamics()
        
        # 3. 控制律
        ddq_ref = ddq_des + self.kp * e + self.kd * de
        tau = M @ ddq_ref + h
        
        return tau