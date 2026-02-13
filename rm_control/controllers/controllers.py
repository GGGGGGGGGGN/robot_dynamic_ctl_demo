import numpy as np

class BaseController:
    """所有控制器的基类（接口定义）"""
    def update(self, q, dq, q_ref, dq_ref, ddq_ref, dynamics_info):
        raise NotImplementedError
        
class JointPDController:
    def __init__(self, kp, kd, pin_dyn=None):
        """
        Args:
            kp (np.array): 比例增益
            kd (np.array): 微分增益
            pin_dyn: PinocchioDynamics 实例。如果为 None，则退化为纯 PD 控制。
        """
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.pin_dyn = pin_dyn
        self.use_comp = pin_dyn is not None
        self.name = "PD + Gravity Comp" if self.use_comp else "Pure PD"

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        """
        现在控制器自己负责从模型获取信息
        """
        # 1. 计算 PD 反馈项
        e = q_ref - q
        de = dq_ref - dq
        tau_pd = self.kp * e + self.kd * de
            
        # 2. 动力学补偿项
        tau_ff = np.zeros_like(q)
        if self.use_comp:
            # 控制器自己负责同步模型状态
            self.pin_dyn.update(q, dq)
            # 获取非线性项 h (重力 + 科氏力)
            _, h = self.pin_dyn.get_dynamics()
            tau_ff = h
        
        return tau_pd + tau_ff

# CTC 控制器同理
class ComputedTorqueController:
    def __init__(self, kp, kd, pin_dyn):
        self.name = "Computed Torque Control (CTC)"
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.pin_dyn = pin_dyn # CTC 必须使用模型

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        self.pin_dyn.update(q, dq)
        M, h = self.pin_dyn.get_dynamics()
        
        e = q_ref - q
        de = dq_ref - dq
        
        # tau = M * (ddq_ref + kp*e + kd*de) + h
        acc_des = ddq_ref + self.kp * e + self.kd * de
        return M @ acc_des + h