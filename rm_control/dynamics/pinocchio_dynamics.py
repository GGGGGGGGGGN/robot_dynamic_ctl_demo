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
        
        # 🔥 [修改点 1] 如果没有传入关节列表，默认使用 Panda 的 7 个关节
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
            # (这意味着 panda_finger_joint1/2 会被选中)
            if jname not in active_joint_names:
                jid = full_model.getJointId(jname)
                joints_to_lock_ids.append(jid)
        
        # 执行裁剪
        if len(joints_to_lock_ids) > 0:
            # 获取参考构型 (将要锁死的关节固定在 0 位置)
            q_ref = pin.neutral(full_model)
            
            # 生成缩减后的模型
            self.model = pin.buildReducedModel(full_model, joints_to_lock_ids, q_ref)
            print(f"✅ 模型裁剪完成! 原自由度: {full_model.nv} -> 现自由度: {self.model.nv}")
        else:
            # 如果白名单包含所有关节，就不裁剪
            self.model = full_model

        # 3. 创建数据结构
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.nq = self.model.nq

        # 4. 确定末端 ID
        # 🔥 [修改点 2] 如果没传 ee_name，默认用 panda_link7，防止报错
        target_ee = ee_name if ee_name else "panda_link7"
        
        if self.model.existFrame(target_ee):
            self.ee_id = self.model.getFrameId(target_ee)
        else:
            # 如果连 panda_link7 都没有，就退化到最后一帧
            self.ee_id = self.model.nframes - 1
            print(f"⚠️ [Pinocchio] 找不到 {target_ee}，使用默认末端: {self.model.frames[self.ee_id].name}")

    def update(self, q, dq):
        """同步状态"""
        if len(q) != self.model.nq:
            print(f"⚠️ [Error] 维度不匹配: 输入q={len(q)}, 模型nq={self.model.nq}")
            return
        # 计算所有的动力学项 (M, h, J 等)
        pin.computeAllTerms(self.model, self.data, q, dq)

    def get_dynamics(self):
        """返回 质量矩阵 M 和 非线性项 h (h = C*dq + g)"""
        return self.data.M.copy(), self.data.nle.copy()

    def get_jacobian(self):
        """获取末端雅可比矩阵 (6 x 7)"""
        # LOCAL_WORLD_ALIGNED: 原点在末端 Link 上，但方向与世界坐标系对齐
        # 这是做 CTC 和 笛卡尔阻抗控制最舒服的坐标系
        J = pin.getFrameJacobian(
            self.model, 
            self.data, 
            self.ee_id, 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J