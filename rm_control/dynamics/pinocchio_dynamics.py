import pinocchio as pin
import numpy as np
import os

class PinocchioDynamics:
    def __init__(self, urdf_path, active_joint_names=None, ee_name="panda_link7"):
        """
        Pinocchio 动力学后端 (默认锁定 Panda 7轴)
        Args:
            urdf_path: URDF 文件路径
            active_joint_names (list):如果不传，默认使用 Panda 的 7 个关节。
            ee_name: 末端执行器名字 (默认 panda_link7)
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"❌ URDF not found: {urdf_path}")

        # 1. 加载完整 URDF 模型 (此时是 9 轴: 7臂 + 2手)
        full_model = pin.buildModelFromUrdf(urdf_path)
        
        # 🔥 如果没有传入关节列表，默认使用 Panda 的 7 个关节
        if active_joint_names is None:
            active_joint_names = [
                "panda_joint1", "panda_joint2", "panda_joint3", 
                "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
            ]
            print(f"ℹ️ [Pinocchio] 未指定关节列表，默认加载 Panda 前 7 轴模式。")

        # 2. 模型裁剪逻辑 (核心)
        # 找出不在 active_joint_names 里的所有关节 ID，准备锁死它们
        joints_to_lock_ids = []
        
        for jname in full_model.names:
            if jname == "universe": continue # 跳过基座
            
            # 如果 URDF 里的关节名字不在我们的白名单里 -> 锁死！
            if jname not in active_joint_names:
                jid = full_model.getJointId(jname)
                joints_to_lock_ids.append(jid)
        
        # 执行裁剪
        if len(joints_to_lock_ids) > 0:
            q_ref = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, joints_to_lock_ids, q_ref)
            print(f"✅ 模型裁剪完成! 原自由度: {full_model.nv} -> 现自由度: {self.model.nv}")
        else:
            self.model = full_model

        # 3. 创建数据结构 (双缓冲机制 🔥)
        self.nv = self.model.nv
        self.nq = self.model.nq
        
        # [A] 主数据：用于存储真实机器人的状态 (q_real)
        self.data = self.model.createData()
        
        # [B] 临时数据：用于计算期望状态或临时查询 (q_ref)，防止污染主数据 🔥
        self.temp_data = self.model.createData()

        # 4. 确定末端 ID
        target_ee = ee_name if ee_name else "panda_link7"
        
        if self.model.existFrame(target_ee):
            self.ee_id = self.model.getFrameId(target_ee)
        else:
            self.ee_id = self.model.nframes - 1
            print(f"⚠️ [Pinocchio] 找不到 {target_ee}，使用默认末端: {self.model.frames[self.ee_id].name}")

    def update(self, q, dq):
        """
        同步机器人的【真实状态】
        注意：这会更新 self.data
        """
        if len(q) != self.model.nq:
            print(f"⚠️ [Error] 维度不匹配: 输入q={len(q)}, 模型nq={self.model.nq}")
            return
        # 计算所有的动力学项 (M, h, J 等) 存入 self.data
        pin.computeAllTerms(self.model, self.data, q, dq)
        pin.computeCoriolisMatrix(self.model, self.data, q, dq)
        
    def compute_forward_kinematics(self, q):
        """
        🔥 新增功能：计算任意关节角 q 下的末端位姿
        
        使用 self.temp_data 进行计算，【绝对不会】影响 update() 里的真实状态。
        适用于：根据 q_ref 计算 x_des
        """
        # 使用 temp_data 计算，安全！
        pin.forwardKinematics(self.model, self.temp_data, q)
        pin.updateFramePlacements(self.model, self.temp_data)
        
        # 获取末端位姿 (SE3 对象)
        pose = self.temp_data.oMf[self.ee_id]
        
        # 返回 位置(3,) 和 旋转矩阵(3,3)
        return pose.translation.copy(), pose.rotation.copy()

    def get_dynamics(self):
        """返回 质量矩阵 M 和 非线性项 h (h = C*dq + g)"""
        return self.data.M.copy(), self.data.nle.copy()

    def get_full_dynamics(self):
        """
        🔥 新增接口：获取完整的动力学“全家桶”，专为动量观测器等高级算法设计
        返回: 
            M: 质量矩阵/惯量矩阵 (7x7)
            C: 科氏力矩阵 (7x7)
            g: 重力向量 (7,)
            h: 非线性项向量 (7,), h = C*dq + g
        """
        # copy() 是为了防止控制器在外部意外修改了底层的缓存数据
        M = self.data.M.copy()
        C = self.data.C.copy()
        g = self.data.g.copy()
        h = self.data.nle.copy()
        
        return M, C, g, h
    
    def get_jacobian(self):
        """获取末端雅可比矩阵 (6 x 7)"""
        J = pin.getFrameJacobian(
            self.model, 
            self.data, 
            self.ee_id, 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J