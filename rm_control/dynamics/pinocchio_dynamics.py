import pinocchio as pin
import numpy as np
import os

class PinocchioDynamics:
    def __init__(self, urdf_path, active_joint_names=None, ee_name=None):
        """
        Pinocchio 动力学后端 (支持模型裁剪)
        Args:
            urdf_path: URDF 文件路径
            active_joint_names (list): [关键] MuJoCo 里存在的关节名字列表。
                                       如果不传，加载完整 URDF。
                                       如果传了，会自动锁死 URDF 里多余的轮子/关节。
            ee_name: 末端执行器名字
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"❌ URDF not found: {urdf_path}")

        # 1. 加载完整 URDF 模型 (包含所有轮子)
        full_model = pin.buildModelFromUrdf(urdf_path)
        
        # 2. 模型裁剪 (Model Reduction)
        if active_joint_names is not None:
            print(f"✂️ [Pinocchio] 收到白名单，正在裁剪模型...")
            
            # 找出需要被锁死的关节 ID
            joints_to_lock_ids = []
            
            # 遍历 URDF 里的所有关节
            for jname in full_model.names:
                if jname == "universe": continue # 跳过宇宙基座
                
                # 如果这个关节不在 MuJoCo 的白名单里，就锁死它！
                if jname not in active_joint_names:
                    # 获取 ID
                    jid = full_model.getJointId(jname)
                    joints_to_lock_ids.append(jid)
                    # print(f"   🔒 锁死冗余关节: {jname}")
            
            if len(joints_to_lock_ids) > 0:
                # 设定被锁死关节的默认位置 (通常是 0)
                q_ref = pin.neutral(full_model)
                
                # 生成缩减后的模型 (只包含 MuJoCo有的关节)
                self.model = pin.buildReducedModel(full_model, joints_to_lock_ids, q_ref)
                print(f"✅ 模型裁剪完成! 原自由度: {full_model.nv} -> 现自由度: {self.model.nv}")
            else:
                print("⚠️ 白名单覆盖了所有关节，无需裁剪。")
                self.model = full_model
        else:
            self.model = full_model

        # 3. 创建数据结构
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.nq = self.model.nq

        # 4. 确定末端 ID
        if ee_name and self.model.existFrame(ee_name):
            self.ee_id = self.model.getFrameId(ee_name)
        else:
            self.ee_id = self.model.nframes - 1
            # print(f"⚠️ [Pinocchio] 未指定 ee_name，默认使用: {self.model.frames[self.ee_id].name}")

    def update(self, q, dq):
        """同步状态"""
        # 简单的维度检查
        if len(q) != self.model.nq:
            print(f"⚠️ [Error] 维度不匹配: MuJoCo q={len(q)}, Pinocchio nq={self.model.nq}")
            return

        pin.computeAllTerms(self.model, self.data, q, dq)

    def get_dynamics(self):
        """返回 M, h"""
        return self.data.M.copy(), self.data.nle.copy()

    def get_jacobian(self):
        """
        获取末端雅可比矩阵 (6 x nv)
        注意：必须在 update() 之后调用
        """
        # 使用 LOCAL_WORLD_ALIGNED (原点在末端，方向对齐世界)
        # 这是做笛卡尔空间控制最常用的 Frame
        J = pin.getFrameJacobian(
            self.model, 
            self.data, 
            self.ee_id, 
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return J