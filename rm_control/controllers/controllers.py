import numpy as np
import pinocchio as pin

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
        self.name = "PD_Gravity_Comp" if self.use_comp else "Pure_PD"

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
        self.name = "Computed_Torque_Control"
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


class ComputedTorqueControllerWithFriction:
    def __init__(self, kp, kd, pin_dyn, fric_coeff=None):
        """
        依赖：M (惯量), h (非线性), Friction (经验模型)
        """
        self.name = "CTC + Friction Comp"
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.pin_dyn = pin_dyn
        
        # 如果没有传入摩擦系数，使用针对 Panda 的经验值
        if fric_coeff is None:
            # J5-J7 需要显著补偿，J1-J4 较小
            self.kv_fric = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
        else:
            self.kv_fric = np.array(fric_coeff)

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        # 1. 模型更新
        self.pin_dyn.update(q, dq)
        M, h = self.pin_dyn.get_dynamics()
        
        # 2. 计算误差
        e = q_ref - q
        de = dq_ref - dq
        
        # 3. 惯性力项 (M * a_des)
        # 即使对于小惯量关节，因为乘了 M，这部分力矩会很小
        acc_des = ddq_ref + self.kp * e + self.kd * de
        tau_inertial = M @ acc_des
        
        # 4. 摩擦力补偿项 (Viscous Friction)
        # 专门对抗 MuJoCo 的 damping，这是 CTC 这种纯刚体动力学缺少的
        tau_fric = self.kv_fric * dq
        
        # 5. 总力矩 = 惯性 + 非线性 + 摩擦
        return tau_inertial + h + tau_fric

# =========================================================
# 2. 笛卡尔空间阻抗控制 (Cartesian Impedance Control)
# =========================================================
class CartesianImpedanceController:
    def __init__(self, kp_cart, kd_cart, pin_dyn):
        """
        控制末端表现得像一个空间弹簧。
        Args:
            kp_cart: 6维刚度 [x, y, z, rx, ry, rz]
            kd_cart: 6维阻尼 [x, y, z, rx, ry, rz]
            pin_dyn: PinocchioDynamics 实例
        """
        self.name = "Cartesian_Impedance"
        self.kp = np.diag(kp_cart)  # (6, 6)
        self.kd = np.diag(kd_cart)  # (6, 6)
        self.pin_dyn = pin_dyn

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        # 1. 更新模型 (计算真实状态 q, dq)
        self.pin_dyn.update(q, dq)
        
        # 获取真实末端位姿 (从主数据 self.data 取)
        J = self.pin_dyn.get_jacobian()
        curr_pose = self.pin_dyn.data.oMf[self.pin_dyn.ee_id]
        p_curr = curr_pose.translation
        R_curr = curr_pose.rotation
        
        # 2. 计算目标末端位姿 (从 q_ref 推算)
        # 🔥 [关键修改] 使用 compute_forward_kinematics (内部用 temp_data)，
        # 绝对不污染主数据的真实状态！
        p_des, R_des = self.pin_dyn.compute_forward_kinematics(q_ref)
        
        # 3. 计算笛卡尔误差 (6维: 3位置 + 3方向)
        # 3.1 位置误差
        err_pos = p_des - p_curr
        
        # 3.2 方向误差 (旋转矩阵差异 -> 转为轴角向量)
        # R_err = R_des * R_curr^T
        # 使用 pin.log3 将旋转矩阵差异转换为 3维误差向量
        R_err = R_des @ R_curr.T
        err_rot = pin.log3(R_err) 
        
        # 合并误差 (6,)
        error = np.concatenate([err_pos, err_rot])
        
        # 4. 计算笛卡尔速度
        v_curr = J @ dq
        v_ref = np.zeros(6) # 简化假设目标静止，或者你需要算 J(q_ref)*dq_ref
        d_error = v_ref - v_curr
        
        # 5. 计算虚拟弹簧力 (Task Space Force)
        # F = Kp * e + Kd * de
        F_task = self.kp @ error + self.kd @ d_error
        
        # 6. 映射回关节力矩
        # tau = J^T * F + h(重力+科氏力)
        # 阻抗控制通常只补偿重力，保留惯性特性
        _, h = self.pin_dyn.get_dynamics()
        
        tau = J.T @ F_task + h
        
        return tau


# =========================================================
# 3. 操作空间控制 (Operational Space Control - OSC)
# =========================================================
class OperationalSpaceController:
    def __init__(self, kp_cart, kd_cart, pin_dyn):
        """
        OSC 试图解耦末端动力学，让末端看起来像一个单位质量的质点。
        计算量比阻抗控制大，但精度通常更高。
        """
        self.name = "Operational_Space_Control"
        self.kp = np.diag(kp_cart)
        self.kd = np.diag(kd_cart)
        self.pin_dyn = pin_dyn

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        # 1. 更新模型
        self.pin_dyn.update(q, dq)
        
        # 获取动力学参数
        M, h = self.pin_dyn.get_dynamics()
        J = self.pin_dyn.get_jacobian()
        
        # 获取真实位姿
        curr_pose = self.pin_dyn.data.oMf[self.pin_dyn.ee_id]
        p_curr = curr_pose.translation
        R_curr = curr_pose.rotation

        # 2. 计算操作空间惯量矩阵 (Lambda)
        # Lambda = (J * M^-1 * J^T)^-1
        # 先求 M 的逆 (对于 7自由度，直接求逆是可以接受的)
        M_inv = np.linalg.inv(M)
        
        # 计算核心项 J * M_inv * J.T
        Lambda_inv = J @ M_inv @ J.T
        
        # 求逆得到 Lambda (添加微小阻尼 1e-4 防止奇异值报错)
        Lambda = np.linalg.inv(Lambda_inv + 1e-4 * np.eye(6))
        
        # 3. 计算目标位姿 (使用安全接口 🔥)
        p_des, R_des = self.pin_dyn.compute_forward_kinematics(q_ref)
        
        # 4. 计算误差 (同上)
        err_pos = p_des - p_curr
        R_err = R_des @ R_curr.T
        err_rot = pin.log3(R_err)
        error = np.concatenate([err_pos, err_rot])
        
        # 速度误差
        v_curr = J @ dq
        d_error = -v_curr # 假设目标不动
        
        # 5. 计算去耦后的控制力 F*
        # OSC 的核心：力 = 惯量 * (期望加速度)
        # acc_des = Kp*e + Kd*de
        acc_cmd = self.kp @ error + self.kd @ d_error
        F_cmd = Lambda @ acc_cmd
        
        # 6. 映射力矩
        # tau = J^T * F_cmd + h
        # (注：严谨的 OSC 还需要 Nullspace 投影来控制手肘姿态，这里为了简化省略)
        tau = J.T @ F_cmd + h
        
        return tau