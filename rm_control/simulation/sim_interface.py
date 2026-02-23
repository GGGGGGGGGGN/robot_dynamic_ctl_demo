import mujoco
import mujoco.viewer
import numpy as np
import time

class SimInterface:
    def __init__(self, xml_path, active_joint_names=None, render=True, dt=0.001, 
                 payload_mass=0.0, payload_offset=[0.0, 0.0, 0.15], payload_size=0.04):
        """
        面向科研的 MuJoCo 仿真接口 (工业级 API 负载注入版)
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
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
        
        self.q_idx = [self.model.jnt_qposadr[i] for i in range(len(self.active_jnt_ids))]
        self.v_idx = [self.model.jnt_dofadr[i] for i in range(len(self.active_jnt_ids))]
        self.nv = len(self.v_idx)
        
        # 2. 识别执行器索引
        self.act_ids = []
        for jid in self.active_jnt_ids:
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid, 0] == jid:
                    self.act_ids.append(aid)
                    
        # 3. 寻找末端执行器 ID
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

        # =====================================================================
        # 魔法核心：API 级物理属性篡改 (Domain Randomization)
        # =====================================================================
        self.payload_mass = payload_mass
        self.payload_offset = np.array(payload_offset)
        self.payload_size = payload_size
        self.ref_path_cache = []

        if self.payload_mass > 0 and self.ee_body_id != -1:
            m0 = self.model.body_mass[self.ee_body_id]
            c0 = self.model.body_ipos[self.ee_body_id].copy()
            m1 = self.payload_mass
            c1 = self.payload_offset
            
            # [Image of center of mass calculation] 计算新的联合质心
            m_new = m0 + m1
            c_new = (m0 * c0 + m1 * c1) / m_new if m_new > 0 else c0
            
            # 暴力写入内存并重置物理常量
            self.model.body_mass[self.ee_body_id] = m_new
            self.model.body_ipos[self.ee_body_id] = c_new
            mujoco.mj_setConst(self.model, self.data)
            
            print(f"📦 动态负载挂载成功！末端总质量从 {m0:.2f}kg 变更为 {m_new:.2f}kg")

        # 4. 渲染配置
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        print(f"🤖 SimInterface 初始化完成. 活跃自由度: {self.nv}")

    def _update_custom_visuals(self):
        """零耗损：极速内存拷贝，构建可视化几何体"""
        if self.viewer is None or not self.viewer.is_running(): return
        self.viewer.user_scn.ngeom = 0 

        # 1. 画轨迹
        for pos in self.ref_path_cache:
            if self.viewer.user_scn.ngeom >= self.viewer.user_scn.maxgeom: break
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.005, 0, 0], 
                pos=pos, mat=np.eye(3).flatten(), rgba=[1.0, 0.0, 0.0, 0.3] 
            )
            self.viewer.user_scn.ngeom += 1

        # 2. 画负载蓝块
        if self.payload_mass > 0:
            if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                ee_pos = self.data.xpos[self.ee_body_id]
                ee_mat = self.data.xmat[self.ee_body_id].reshape(3, 3)
                payload_global_pos = ee_pos + ee_mat @ self.payload_offset

                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[self.payload_size, self.payload_size, self.payload_size],
                    pos=payload_global_pos, mat=ee_mat.flatten(), rgba=[0.1, 0.5, 0.8, 0.9]
                )
                self.viewer.user_scn.ngeom += 1

    def render(self):
        """统一的渲染接口，取代单纯的 viewer.sync()"""
        if self.viewer is not None and self.viewer.is_running():
            self._update_custom_visuals()
            self.viewer.sync()

    def step(self):
        """仿真步进，并自动处理视觉同步"""
        mujoco.mj_step(self.model, self.data)
        self.render()

    def set_control_mode(self, mode="torque"):
        max_torques = [87, 87, 87, 87, 12, 12, 12]
        for aid in self.act_ids:
            if mode == "torque":
                limit = max_torques[aid] if aid < 7 else 12 
                self.model.actuator_ctrlrange[aid] = [-limit, limit]
                self.model.actuator_forcerange[aid] = [-limit, limit]
                self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE 
                self.model.actuator_gainprm[aid, 0] = 1.0     
                self.model.actuator_biasprm[aid, :3] = 0.0    
            elif mode == "position":
                self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE 
                self.model.actuator_gainprm[aid, 0] = 4500.0  
                self.model.actuator_biasprm[aid, :3] = [0.0, -4500.0, -450.0] 

    def get_state(self):
        return self.data.qpos[self.q_idx].copy(), self.data.qvel[self.v_idx].copy()

    def set_joint_torque(self, tau):
        self.data.ctrl[self.act_ids] = tau

    def is_alive(self):
        return self.viewer.is_running() if self.viewer else True

    def close(self):
        if self.viewer: self.viewer.close()

    def calc_fk(self, q_target):
        q_backup = self.data.qpos[:7].copy()
        self.data.qpos[:7] = q_target
        mujoco.mj_kinematics(self.model, self.data) 
        pos = self.data.xpos[self.ee_body_id].copy()
        self.data.qpos[:7] = q_backup
        mujoco.mj_kinematics(self.model, self.data) 
        return pos

    def precompute_trajectory(self, traj_generator):
        self.ref_path_cache = []
        for i, t in enumerate(traj_generator.time_steps):
            if i % 20 == 0:
                q_ref, _, _ = traj_generator.get_state(t)
                self.ref_path_cache.append(self.calc_fk(q_ref))