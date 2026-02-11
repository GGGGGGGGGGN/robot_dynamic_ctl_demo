import numpy as np
import time
import mujoco

# 引入我们写好的模块
from rm_control.simulation.sim_interface import SimInterface
from rm_control.planning.trajectory import TrajectoryGenerator

# ==========================================
# 1. 动力学服务类 (封装 MuJoCo 计算)
# ==========================================
class DynamicsServer:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nv = model.nv
        # 预分配内存
        self.M = np.zeros((self.nv, self.nv))

    def get_dynamics(self):
        # 计算质量矩阵 M (稠密矩阵)
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        # 计算非线性项 h = 重力 + 科里奥利 + 离心力
        h = self.data.qfrc_bias.copy()
        return self.M, h

# ==========================================
# 2. CTC 控制器类
# ==========================================
class CTCController:
    def __init__(self, dyn_server, kp, kd):
        self.dyn = dyn_server
        self.kp = np.array(kp)
        self.kd = np.array(kd)

    def compute(self, q_curr, dq_curr, q_des, dq_des, ddq_des):
        # 1. 计算误差
        e = q_des - q_curr
        de = dq_des - dq_curr
        
        # 2. 获取动力学 M, h
        M, h = self.dyn.get_dynamics()
        
        # 3. 计算期望加速度 (PD 反馈 + 前馈)
        ddq_ref = ddq_des + self.kp * e + self.kd * de
        
        # 4. 动力学方程: tau = M * ddq_ref + h
        tau = M @ ddq_ref + h
        
        return tau

# ==========================================
# 3. 主函数
# ==========================================
def main():
    # ⚠️ 必须使用 'torque' 模式
    sim = SimInterface(mode='torque', render=True)
    
    # 初始化工具
    dyn_server = DynamicsServer(sim.model, sim.data)
    traj_gen = TrajectoryGenerator()
    
    # --- 调参区域 ---
    # CTC 使得系统变成线性二阶系统，Kp可以给大一点
    kp_value = 100.0
    kd_value = 2.0 * np.sqrt(kp_value) # 临界阻尼公式
    
    ctc = CTCController(dyn_server, 
                        kp=[kp_value] * sim.nv, 
                        kd=[kd_value] * sim.nv)
    
    # --- 轨迹定义 ---
    # 定义两个点：Home(全0) 和 Target(举手)
    q_home = np.zeros(sim.nv)
    q_target = np.zeros(sim.nv)
    
    # 找到左臂在全局向量中的索引位置
    left_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_left]
    # 设置左臂目标：抬起，弯曲肘部
    q_target[left_indices] = np.array([0, -0.5, 1.5, 0.5, 1.0, 0])
    
    # 运动参数
    motion_duration = 2.0  # 单程 2 秒
    wait_time = 0.5        # 到达后停顿 0.5 秒
    
    # 状态机变量
    start_time = sim.get_time()
    current_start_q = q_home
    current_end_q = q_target
    is_moving_forward = True # 标记方向

    print("🚀 CTC 轨迹跟踪开始！机器人将在两点间往复运动...")

    while sim.is_alive():
        # 1. 获取时间
        t_curr = sim.get_time()
        t_rel = t_curr - start_time
        
        # 2. 轨迹规划 (核心)
        # 自动计算当前时刻这一毫秒应该在什么角度、速度、加速度
        q_des, dq_des, ddq_des = traj_gen.min_jerk(
            current_start_q, current_end_q, motion_duration, t_rel
        )
        
        # --- 逻辑：切换目标点 (往复运动) ---
        if t_rel > (motion_duration + wait_time):
            start_time = t_curr
            # 交换起点和终点
            current_start_q, current_end_q = current_end_q, current_start_q
            print(f"🔄 切换方向: {'去目标点' if not is_moving_forward else '回原点'}")
            is_moving_forward = not is_moving_forward
            
        # 3. 获取机器人真实状态
        q_now, dq_now = sim.get_state()
        
        # 4. CTC 计算力矩 (全量计算)
        tau_full = ctc.compute(q_now, dq_now, q_des, dq_des, ddq_des)
        
        # 5. 分发力矩 (千万别忘了头部和右臂！)
        # 左臂：跟随轨迹
        sim.set_left_arm_cmd(tau_full[left_indices])
        
        # 右臂：保持在 0 位 (抗重力)
        right_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_right]
        sim.set_right_arm_cmd(tau_full[right_indices])
        
        # 头部：保持在 0 位 (抗重力，防止掉头)
        head_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_head]
        sim.set_head_cmd(tau_full[head_indices])
        
        # 升降台：保持在 0 位
        plat_indices = [sim.model.jnt_qposadr[i] for i in sim.idx_jnt_platform]
        sim.set_platform_cmd(tau_full[plat_indices])
        
        # 6. 物理步进
        sim.step()
        
        # 简单控制一下帧率，防止 Python 跑太快看不清
        time.sleep(sim.dt)

if __name__ == "__main__":
    main()