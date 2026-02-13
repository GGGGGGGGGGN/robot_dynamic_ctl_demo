import mujoco
import mujoco.viewer
import numpy as np
import time

class SimInterface:
    def __init__(self, xml_path, active_joint_names=None, render=True, dt=0.001):
        """
        面向科研的 MuJoCo 仿真接口
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 默认步长
        self.dt = dt 
        self.model.opt.timestep = dt
        
        # 1. 自动识别关节索引
        if active_joint_names is None:
            self.active_jnt_ids = [i for i in range(self.model.njnt) 
                                 if self.model.jnt_type[i] == 3
                                 or self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_SLIDE]
        else:
            self.active_jnt_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                                 for name in active_joint_names]
        
        # 获取对应的 qpos 和 qvel 索引
        self.q_idx = [self.model.jnt_qposadr[i] for i in range(len(self.active_jnt_ids))]
        self.v_idx = [self.model.jnt_dofadr[i] for i in range(len(self.active_jnt_ids))]
        self.nv = len(self.v_idx)
        
        # 2. 识别执行器索引
        self.act_ids = []
        for jid in self.active_jnt_ids:
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid, 0] == jid:
                    self.act_ids.append(aid)
        
        # ----------------------------------------------------------------------
        # 🔥 [修复核心] 自动寻找末端执行器 ID (用于 FK 计算和画图)
        # ----------------------------------------------------------------------
        possible_names = ["panda_link7", "link7", "panda_hand", "hand", "end_effector"]
        self.ee_body_id = -1
        
        for name in possible_names:
            try:
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if bid >= 0:
                    self.ee_body_id = bid
                    print(f"✅ 锁定末端执行器: {name} (ID: {self.ee_body_id})")
                    break
            except:
                continue
        
        if self.ee_body_id == -1:
            self.ee_body_id = self.model.nbody - 1
            print(f"⚠️ 未找到指定末端，默认使用最后一个 Body (ID: {self.ee_body_id})")

        # 🔥 [修复核心] 初始化轨迹缓存
        self.ref_path_cache = []
        # ----------------------------------------------------------------------

        # 3. 渲染配置
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        print(f"🤖 SimInterface 初始化完成. 活跃自由度: {self.nv}")

    def set_control_mode(self, mode="torque"):
        """
        动态切换执行器模式
        """
        max_torques = [87, 87, 87, 87, 12, 12, 12]
        for aid in self.act_ids:
            if mode == "torque":
                limit = max_torques[aid] if aid < 7 else 12 # 防止数组越界，简单保护
                self.model.actuator_ctrlrange[aid] = [-limit, limit]
                self.model.actuator_forcerange[aid] = [-limit, limit]
                self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE 
                self.model.actuator_gainprm[aid, 0] = 1.0     
                self.model.actuator_biasprm[aid, :3] = 0.0    
            elif mode == "position":
                self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE 
                self.model.actuator_gainprm[aid, 0] = 4500.0  
                self.model.actuator_biasprm[aid, :3] = [0.0, -4500.0, -450.0] 
        print(f"🛠️  模式切换至: {mode.upper()}")

    def get_state(self):
        """返回当前活跃关节的位置和速度"""
        q = self.data.qpos[self.q_idx].copy()
        dq = self.data.qvel[self.v_idx].copy()
        return q, dq

    def set_joint_torque(self, tau):
        """发送力矩指令"""
        if len(tau) == len(self.act_ids):
            self.data.ctrl[self.act_ids] = tau
        else:
            raise ValueError("力矩维度与执行器数量不匹配")

    def step(self):
        """仿真步进"""
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def is_alive(self):
        if self.viewer is not None:
            return self.viewer.is_running()
        return True

    def get_physics_params(self):
        dampings = self.model.dof_damping[self.v_idx].copy()
        armatures = self.model.dof_armature[self.v_idx].copy()
        return dampings, armatures

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
        print("🔄 SimInterface: Environment Reset.")
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    # ==========================================================================
    # FK 和 轨迹相关函数
    # ==========================================================================
    def calc_fk(self, q_target):
        """传入关节角 q，计算对应的末端笛卡尔坐标 (x, y, z)"""
        q_backup = self.data.qpos[:7].copy()
        self.data.qpos[:7] = q_target
        mujoco.mj_kinematics(self.model, self.data) # 只算运动学
        
        # 这里之前报错，因为 ee_body_id 没定义，现在修复了
        pos = self.data.xpos[self.ee_body_id].copy()
        
        self.data.qpos[:7] = q_backup
        mujoco.mj_kinematics(self.model, self.data) # 恢复
        return pos

    def precompute_trajectory(self, traj_generator):
        """预计算轨迹可视化点"""
        print("🔄 正在预计算参考轨迹可视化路径...")
        # 这里之前可能也会报错，因为 ref_path_cache 没定义，现在修复了
        self.ref_path_cache = []
        
        downsample_rate = 20 
        for i, t in enumerate(traj_generator.time_steps):
            if i % downsample_rate == 0:
                q_ref, _, _ = traj_generator.get_state(t)
                pos = self.calc_fk(q_ref)
                self.ref_path_cache.append(pos)
                
        print(f"✅ 轨迹预计算完成，共缓存 {len(self.ref_path_cache)} 个可视化点。")

    def draw_trajectory(self, viewer):
        """在 MuJoCo viewer 里绘制参考轨迹"""
        if not self.ref_path_cache:
            return

        if viewer.user_scn.ngeom + len(self.ref_path_cache) >= viewer.user_scn.maxgeom:
            viewer.user_scn.ngeom = 0 
            
        for pos in self.ref_path_cache:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0], 
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 0.3] 
            )
            viewer.user_scn.ngeom += 1