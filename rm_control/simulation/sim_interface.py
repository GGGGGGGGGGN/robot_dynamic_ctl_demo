import mujoco
import mujoco.viewer
import numpy as np
import time

class SimInterface:
    def __init__(self, xml_path, active_joint_names=None, render=True):
        """
        面向科研的 MuJoCo 仿真接口
        :param xml_path: MJCF 文件路径
        :param active_joint_names: 活跃关节名称列表（如 Panda 的 7 个关节）
        :param render: 是否开启渲染
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 默认步长
        self.dt = self.model.opt.timestep
        
        # 1. 自动识别关节索引
        if active_joint_names is None:
            # 如果不指定，默认取所有 1 自由度关节
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
        # 假设执行器与活跃关节是一一对应的
        self.act_ids = []
        for jid in self.active_jnt_ids:
            for aid in range(self.model.nu):
                if self.model.actuator_trnid[aid, 0] == jid:
                    self.act_ids.append(aid)
        
        # 3. 渲染配置
        self.viewer = None
        if render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        print(f"🤖 SimInterface 初始化完成. 活跃自由度: {self.nv}")

    def set_control_mode(self, mode="torque"):
        """
        动态切换执行器模式
        :param mode: "torque" (纯力矩, gain=1, bias=none) 或 "position" (XML 默认 PD)
        """
        max_torques = [87, 87, 87, 87, 12, 12, 12]
        for aid in self.act_ids:
            if mode == "torque":
                limit = max_torques[aid]
                # 设置为真实力矩极限
                self.model.actuator_ctrlrange[aid] = [-limit, limit]
                self.model.actuator_forcerange[aid] = [-limit, limit]
                self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_NONE # 禁用偏置项 
                self.model.actuator_gainprm[aid, 0] = 1.0     # 增益设为 1 
                self.model.actuator_biasprm[aid, :3] = 0.0    # 清零 PD 参数 
            elif mode == "position":
                # 回复到 Panda XML 的默认 PD 设置 
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
        """发送力矩指令 (仅在 torque 模式下有效)"""
        # 注意：mu_coulomb 和其他摩擦在 mj_inverse 中体现，此处仅设置控制输入 
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
        """获取 XML 中定义的阻尼和电枢惯量，用于控制器补偿"""
        # damping 对应 XML 中的 joint damping 
        dampings = self.model.dof_damping[self.v_idx].copy()
        # armature 对应 XML 中的 joint armature 
        armatures = self.model.dof_armature[self.v_idx].copy()
        return dampings, armatures

    def reset(self):
        """重置仿真环境到初始状态"""
        # 1. 重置数据 (qpos, qvel 等恢复到 XML 定义的初始值)
        mujoco.mj_resetData(self.model, self.data)
        
        # 2. 必须手动调用一次前向运动学，确保 xpos, xquat 等派生数据同步更新
        mujoco.mj_forward(self.model, self.data)
        
        # 3. 如果之前有 viewer，有时需要刷新一下
        if self.viewer:
            self.viewer.sync()
            
        print("🔄 SimInterface: Environment Reset.")
        
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            
            
    def calc_fk(self, q_target):
        """
        传入关节角 q，计算对应的末端笛卡尔坐标 (x, y, z)
        """
        # 1. 备份当前物理状态
        q_backup = self.data.qpos[:7].copy()
        
        # 2. 瞬移到目标姿态
        self.data.qpos[:7] = q_target
        
        # 3. 仅计算运动学 (Kinematics)，不计算动力学
        mujoco.mj_kinematics(self.model, self.data)
        
        # 4. 获取位置
        pos = self.data.xpos[self.ee_body_id].copy()
        
        # 5. 恢复现场 (非常重要！否则物理仿真会瞬变)
        self.data.qpos[:7] = q_backup
        mujoco.mj_kinematics(self.model, self.data) # 恢复缓存
        
        return pos

    # ==========================================================================
    # 🔥 新增功能 2: 预计算并缓存整条轨迹
    # ==========================================================================
    def precompute_trajectory(self, traj_generator):
        """
        接收轨迹生成器，计算出所有时间点的末端位置，存入 cache
        """
        print("🔄 正在预计算参考轨迹可视化路径...")
        self.ref_path_cache = []
        
        # 遍历时间步 (使用生成器里的 time_steps)
        # 为了画图不卡顿，我们每隔 10 个点采一个样 (降采样)
        downsample_rate = 20 
        
        for i, t in enumerate(traj_generator.time_steps):
            if i % downsample_rate == 0:
                # 1. 拿到关节空间目标 q
                q_ref, _, _ = traj_generator.get_state(t)
                
                # 2. 算出笛卡尔空间位置 xyz
                pos = self.calc_fk(q_ref)
                
                self.ref_path_cache.append(pos)
                
        print(f"✅ 轨迹预计算完成，共缓存 {len(self.ref_path_cache)} 个可视化点。")

    # ==========================================================================
    # 🔥 新增功能 3: 在 Viewer 里画出来
    # ==========================================================================
    def draw_trajectory(self, viewer):
        """
        在 MuJoCo viewer 里绘制参考轨迹 (红色面包屑)
        """
        # 如果缓存是空的，就不画
        if not self.ref_path_cache:
            return

        # 检查 Geom 数量是否超限
        if viewer.user_scn.ngeom + len(self.ref_path_cache) >= viewer.user_scn.maxgeom:
            viewer.user_scn.ngeom = 0 # 满了就清空重画
            
        # 遍历缓存的点，画红色小球
        for pos in self.ref_path_cache:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],  # 半径 5mm 的小球
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 0.3] # 红色，透明度 0.3 (淡淡的虚影)
            )
            viewer.user_scn.ngeom += 1
            
    def draw_trajectory(self, viewer):
        """
        在 MuJoCo viewer 里绘制参考轨迹 (红色虚线/面包屑)
        """
        # 如果缓存是空的，就不画
        if not self.ref_path_cache:
            return

        # 检查 Geom 数量是否接近上限 (MuJoCo 默认上限比较低)
        # 如果满了，就不添加新的，或者清空重画
        if viewer.user_scn.ngeom + len(self.ref_path_cache) >= viewer.user_scn.maxgeom:
            viewer.user_scn.ngeom = 0 
            
        # 遍历缓存的点，画红色小球
        for pos in self.ref_path_cache:
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.005, 0, 0],  # 半径 5mm 的小球
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=[1.0, 0.0, 0.0, 0.3] # 红色，透明度 0.3 (半透明，看着高级)
            )
            viewer.user_scn.ngeom += 1