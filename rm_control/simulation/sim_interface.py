
from numpy.lib._format_impl import EXPECTED_KEYS
import numpy as np
import mujoco
import mujoco.viewer

class SimInterface:
    def __init__(self, model_path, render=True):
        self.model_path = model_path
        self.render = render
        if not self.model_path:
            raise ValueError("❌ 必须提供模型路径！SimInterface 不再自带模型了。")
        print(f"🔄 SimInterface 正在加载: {self.model_path}")
        try:
            if self.model_path.endswith(".xml"):
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
            elif self.model_path.endswith(".mjb"):
                self.model = mujoco.MjModel.from_binary_path(self.model_path)
                
            self.data = mujoco.MjData(self.model)

            self.control_mode = self._detect_control_mode()
            print(f"🤖 自动识别控制模式: {self.control_mode}")
        except ValueError as e:
            print(f"❌ 模型加载失败: {e}")
            raise

        # --- 2. 获取基本维度 ---
        self.dt = self.model.opt.timestep
        self.nu = self.model.nu  # 执行器数量
        self.nq = self.model.nq  # 关节位置维度
        self.nv = self.model.nv  # 关节速度维度

        # --- 3. 初始化控制缓存 ---
        # 维护一个全量的控制数组，分部控制函数只更新这个数组的一部分
        self.current_ctrl = np.zeros(self.nu)

        # --- 4. 建立索引映射 (关键步骤) ---
        self._init_indices()

        print(f"✅ 模型加载成功！模式: {self.control_mode.upper()}, Actuators: {self.nu}")
        
        # --- 5. 启动 Viewer ---
        self.viewer = None
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("🖥️  图形界面已启动")

    def _init_indices(self):
        """
        [修复版] 根据 XML 中的命名规则，自动找到各部位对应的索引。
        """
        actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
                          for i in range(self.nu)]
        
        # 1. 执行器索引 (使用更严格的匹配 'act_l' 和 'act_r')
        # 这样 'act_platform' 就不会因为包含 'l' 或 'r' 而被误判了
        self.idx_act_left = [i for i, n in enumerate(actuator_names) if 'act_l' in n]
        self.idx_act_right = [i for i, n in enumerate(actuator_names) if 'act_r' in n]
        
        # 头部和升降台保持不变
        self.idx_act_head = [i for i, n in enumerate(actuator_names) if 'head' in n]
        self.idx_act_platform = [i for i, n in enumerate(actuator_names) if 'platform' in n]

        # 2. 关节位置索引 (同理修复)
        joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                       for i in range(self.model.njnt)]
        
        # 假设关节命名是 'l_joint1', 'r_joint1' 等
        self.idx_jnt_left = [i for i, n in enumerate(joint_names) if 'l_joint' in n]
        self.idx_jnt_right = [i for i, n in enumerate(joint_names) if 'r_joint' in n]
        self.idx_jnt_head = [i for i, n in enumerate(joint_names) if 'head' in n]
        self.idx_jnt_platform = [i for i, n in enumerate(joint_names) if 'platform' in n]
        
        # 打印调试信息 (这样你就能看到现在是 6 个了)
        print(f"🔍 索引映射结果:")
        print(f"   - 左臂执行器ID (Count: {len(self.idx_act_left)}): {self.idx_act_left}")
        print(f"   - 右臂执行器ID (Count: {len(self.idx_act_right)}): {self.idx_act_right}")
        print(f"   - 升降台执行器ID: {self.idx_act_platform}")

    # =========================================================================
    #                               核心控制接口
    # =========================================================================

    def step(self):
        """
        执行一步仿真。
        注意：不再需要传入 action 参数，而是直接使用内部维护的 self.current_ctrl
        """
        # 1. 写入控制指令
        self.data.ctrl[:] = self.current_ctrl
        
        # 2. 物理步进
        mujoco.mj_step(self.model, self.data)
        
        # 3. 渲染
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    # =========================================================================
    #                               分部控制 Setter
    # =========================================================================

    def set_left_arm_cmd(self, cmd):
        """设置左臂指令 (Pos/Torque)"""
        if len(cmd) != len(self.idx_act_left):
            print(f"⚠️ 左臂维度错误: 需要 {len(self.idx_act_left)}, 收到 {len(cmd)}")
            return
        self.current_ctrl[self.idx_act_left] = cmd

    def set_right_arm_cmd(self, cmd):
        """设置右臂指令 (Pos/Torque)"""
        if len(cmd) != len(self.idx_act_right):
            print(f"⚠️ 右臂维度错误: 需要 {len(self.idx_act_right)}, 收到 {len(cmd)}")
            return
        self.current_ctrl[self.idx_act_right] = cmd

    def set_head_cmd(self, cmd):
        """设置头部指令"""
        self.current_ctrl[self.idx_act_head] = cmd

    def set_platform_cmd(self, cmd):
        """设置升降台指令"""
        self.current_ctrl[self.idx_act_platform] = cmd

    def set_whole_body_cmd(self, cmd):
        """设置全身指令 (兼容旧接口)"""
        if len(cmd) != self.nu:
            return
        self.current_ctrl[:] = cmd

    # =========================================================================
    #                               分部状态 Getter
    # =========================================================================

    def get_state(self):
        """
        获取机器人整体状态 (全量)
        
        Returns:
            qpos (np.array): 整体关节位置 (维度 nq)
            qvel (np.array): 整体关节速度 (维度 nv)
        """
        # 必须使用 .copy()，否则返回的是指针，数据会在计算过程中突变
        return self.data.qpos.copy(), self.data.qvel.copy()
    
    
    def get_left_arm_qpos(self):
        """获取左臂关节角度"""
        # qpos 的索引可能与 joint 索引需要通过 jnt_qposadr 转换，
        # 但对于简单转动关节，通常是直接映射的。严谨做法如下：
        indices = [self.model.jnt_qposadr[i] for i in self.idx_jnt_left]
        return self.data.qpos[indices]

    def get_right_arm_qpos(self):
        """获取右臂关节角度"""
        indices = [self.model.jnt_qposadr[i] for i in self.idx_jnt_right]
        return self.data.qpos[indices]

    def get_time(self):
        return self.data.time

    def is_alive(self):
        if self.render and self.viewer:
            return self.viewer.is_running()
        return True

    def close(self):
        if self.viewer:
            self.viewer.close()


    def _detect_control_mode(self):
            """
            智能判别模式：文件名优先 -> 物理属性兜底
            """
            # === 策略 1: 检查文件名 (最稳) ===
            # 既然你有两个文件，通常一个叫 scene_torque.xml，一个叫 scene_pos.xml
            path_str = self.model_path.lower()
            if "torque" in path_str:
                return "torque"
            if "pos" in path_str or "joint" in path_str:
                return "position"
            else:
                return "unknown"