import time
import os
import numpy as np
import mujoco
import mujoco.viewer

from rm_control.assets import get_model_path_torque
from rm_control.assets import get_model_path_position


class SimInterface:
    def __init__(self, mode='position', render=True):
        """
        初始化仿真接口
        
        Args:
            mode (str): 'position' (位置控制) 或 'torque' (力矩控制)
            render (bool): 是否开启图形界面 (GUI)
        """
        self.render = render
        self.mode = mode
        
        # 1. 自动定位 XML 路径
        # 假设 assets 文件夹在 simulation 文件夹的上一级
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) # 回退一级
        
        if mode == 'position':
            xml_path = get_model_path_position()
        elif mode == 'torque':
            xml_path = get_model_path_torque()
        else:
            raise ValueError(f"未知模式: {mode}, 请使用 'position' 或 'torque'")

        print(f"📖 [SimInterface] 正在加载模型: {xml_path}")
        
        # 2. 加载模型
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except ValueError as e:
            print(f"❌ 模型加载失败，请检查路径！\n错误信息: {e}")
            raise

        # 3. 获取基本信息
        self.dt = self.model.opt.timestep
        self.nu = self.model.nu  # 执行器数量 (Actuators)
        self.nq = self.model.nq  # 关节位置维度
        self.nv = self.model.nv  # 关节速度维度
        
        # 获取执行器名字列表，方便调试
        self.actuator_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) 
                               for i in range(self.nu)]
        
        print(f"✅ 模型加载成功！\n"
              f"   - 模式: {mode.upper()}\n"
              f"   - 执行器数量: {self.nu}\n"
              f"   - 时间步长: {self.dt}s")

        # 4. 初始化 Viewer (被动模式，非阻塞)
        self.viewer = None
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("🖥️  图形界面 (GUI) 已启动")
        else:
            print("🚫 图形界面 (GUI) 已关闭 (Headless Mode)")

    def step(self, action):
        """
        仿真推演一步
        
        Args:
            action (np.array): 控制指令
                               - Position模式: 目标角度 (rad)
                               - Torque模式:   目标力矩 (Nm)
        """
        # 安全检查：维度必须匹配
        if len(action) != self.nu:
            print(f"⚠️ 警告: 输入维度 {len(action)} 不等于执行器数量 {self.nu}")
            return

        # 1. 写入控制指令
        self.data.ctrl[:] = action
        
        # 2. 物理引擎计算
        # 通常物理频率比控制频率高，这里演示 1:1，实际可能需要循环多次 mj_step
        mujoco.mj_step(self.model, self.data)
        
        # 3. 更新画面 (如果开启)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def get_state(self):
        """
        获取机器人当前状态
        Returns:
            qpos (np.array): 关节位置
            qvel (np.array): 关节速度
        """
        return self.data.qpos.copy(), self.data.qvel.copy()

    def get_time(self):
        """获取当前仿真时间"""
        return self.data.time

    def reset(self):
        """重置仿真环境"""
        mujoco.mj_resetData(self.model, self.data)
        if self.viewer:
            self.viewer.sync()
            
    def is_alive(self):
        """检查 Viewer 是否还活着 (如果关闭了窗口，仿真也应该停止)"""
        if self.render and self.viewer:
            return self.viewer.is_running()
        return True

    def close(self):
        """关闭环境"""
        if self.viewer:
            self.viewer.close()