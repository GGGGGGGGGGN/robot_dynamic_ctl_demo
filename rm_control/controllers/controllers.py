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
    def __init__(self, kp, kd, pin_dyn, kv_fric=None, kc_fric=None):
        """
        升级版：包含 粘性摩擦(kv) 和 库仑/静摩擦(kc) 补偿的 CTC
        """
        self.name = "CTC + Full Friction Comp"
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.pin_dyn = pin_dyn
        
        # 粘性摩擦系数 (与速度成正比)
        self.kv_fric = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5]) if kv_fric is None else np.array(kv_fric)
        # 库仑摩擦系数 (恒定阻力，只与速度方向有关)
        self.kc_fric = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]) if kc_fric is None else np.array(kc_fric)

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        self.pin_dyn.update(q, dq)
        M, h = self.pin_dyn.get_dynamics()
        
        # 🔥 神级修复：强行把 MuJoCo XML 里隐藏的 armature (0.1) 补回到质量矩阵里！
        # 因为 XML 里每个关节的 armature 都是 0.1
        M_real = M + np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        e = q_ref - q
        de = dq_ref - dq
        
        acc_des = ddq_ref + self.kp * e + self.kd * de
        tau_inertial = M_real @ acc_des
        
        # 🔥 工业级摩擦力补偿：粘性摩擦 (kv * dq) + 库仑摩擦 (kc * sign(dq))
        # np.sign(dq) 会提取速度的方向 (1, -1, 或 0)
        tau_fric = self.kv_fric * dq + self.kc_fric * np.sign(dq)
        
        return tau_inertial + h + tau_fric




class PIDComputedTorqueController:
    def __init__(self, kp, ki, kd, pin_dyn, kv_fric=None, kc_fric=None, dt=0.001, integral_limit=10.0):
        """
        终极兜底版：包含 摩擦补偿 + 积分项(I) + 抗积分饱和(Anti-windup) 的 PID-CTC 控制器
        """
        self.name = "PID-CTC + Friction Comp"
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.ki = np.array(ki)  # 🔥 新增：积分增益参数
        self.pin_dyn = pin_dyn
        self.dt = dt            # 积分需要用到时间步长
        
        # 🔥 新增：误差积分累加器，初始为0
        self.error_sum = np.zeros(7)
        
        # 🔥 新增：抗积分饱和限制 (极其重要！)
        # 限制积分项最多只能提供一定数值的加速度补偿，防止误差爆炸
        self.integral_limit = np.array(integral_limit) if isinstance(integral_limit, (list, np.ndarray)) else np.ones(7) * integral_limit
        
        # 粘性摩擦与库仑摩擦系数 (保持不变)
        self.kv_fric = np.array([0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5]) if kv_fric is None else np.array(kv_fric)
        self.kc_fric = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2]) if kc_fric is None else np.array(kc_fric)

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        self.pin_dyn.update(q, dq)
        M, h = self.pin_dyn.get_dynamics()
        
        # 补偿 MuJoCo XML 里隐藏的 armature (0.1)
        M_real = M + np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        e = q_ref - q
        de = dq_ref - dq
        
        # =========================================================
        # 🔥 核心魔法：积分项与抗饱和机制
        # =========================================================
        # 1. 累加误差 (数值积分: error * dt)
        self.error_sum += e * self.dt
        
        # 2. 抗积分饱和 (Clamping/Anti-windup)：强行把累加的误差限制在安全范围内
        self.error_sum = np.clip(self.error_sum, -self.integral_limit, self.integral_limit)
        
        # 3. 计算期望加速度 (引入了 ki * error_sum)
        acc_des = ddq_ref + self.kp * e + self.kd * de + self.ki * self.error_sum
        # =========================================================

        # 动力学前馈补偿
        tau_inertial = M_real @ acc_des
        
        # 摩擦力补偿
        tau_fric = self.kv_fric * dq + self.kc_fric * np.sign(dq)
        
        return tau_inertial + h + tau_fric
    


class MomentumObserverCTC:
    def __init__(self, kp, kd, ko, pin_dyn, dt=0.001, tau_clip = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0]):
        """
        基于动量观测器 (MOB) 的鲁棒计算力矩控制器
        :param ko: 观测器带宽增益 (例如 [50, 50, 50...])
        """
        self.name = "MOB-CTC"
        self.kp = np.array(kp)
        self.kd = np.array(kd)
        self.ko = np.diag(ko)  # 观测器增益矩阵
        self.pin_dyn = pin_dyn
        self.dt = dt
        self.tau_clip = np.array(tau_clip)
        
        # 内部状态记忆
        self.p_hat = np.zeros(7)       # 虚拟动量估计
        self.r = np.zeros(7)           # 极其重要的残差 r (估算出的未知负载/摩擦力)
        self.last_tau_cmd = np.zeros(7) # 记录上一毫秒下发的指令力矩
        self.is_initialized = False

    def update(self, q, dq, q_ref, dq_ref, ddq_ref):
        # 1. 更新物理底座，并获取完整的动力学“全家桶”
        self.pin_dyn.update(q, dq)
        M, C, g, h = self.pin_dyn.get_full_dynamics()
        
        # 依然需要补回 MuJoCo 隐藏的 armature 惯量
        M_real = M + np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 
        
        # =========================================================
        # ⚡ 动量观测器核心 (干净无依赖)
        # =========================================================
        p = M_real @ dq  # 当前真实动量
        
        if not self.is_initialized:
            self.p_hat = p.copy()
            self.is_initialized = True
            
        # 计算残差 r (外部扰动)
        self.r = self.ko @ (p - self.p_hat)
        
        # 动量估计更新 (数值积分)
        dp_hat = self.last_tau_cmd + C.T @ dq - g + self.r
        self.p_hat += dp_hat * self.dt
        # =========================================================

        # 2. 计算标准 CTC 前馈 + PD 反馈
        e = q_ref - q
        de = dq_ref - dq
        acc_des = ddq_ref + self.kp * e + self.kd * de
        tau_nominal = M_real @ acc_des + h
        
        # 3. 终极一击：名义力矩 - 观测到的扰动
        tau_cmd = tau_nominal - self.r
        
        # 记忆当前下发的力矩，供下一毫秒的观测器使用      
        tau_actual = np.clip(tau_cmd, -self.tau_clip, self.tau_clip)
        
        # 极其致命的一步：把【真实执行的力矩】记入账本，喂给下一帧的观测器！
        self.last_tau_cmd = tau_actual
        
        return tau_actual
    
    
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