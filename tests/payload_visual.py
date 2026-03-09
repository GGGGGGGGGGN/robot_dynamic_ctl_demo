import time
import numpy as np
import mujoco

# 导入你刚刚完美修复的 SimInterface
from rm_control.simulation.sim_interface import SimInterface

def main():
    # 1. 配置文件路径 (请根据你的实际路径调整)
    xml_path = "rm_control/assets/franka_emika_panda/scene.xml"  
    
    print("启动仿真引擎并挂载负载...")
    
    # 2. 实例化仿真器，挂载 5kg 铁块
    sim = SimInterface(
        xml_path=xml_path, 
        render=True, 
        dt=0.001,
        payload_mass=5.0,              
        # Z轴正方向是夹爪向外的方向。0.15m (15厘米) 大概在夹爪指尖
        payload_offset=[0.0, 0.0, 0.0], 
        payload_size=0.04 # 这是一个半边长，真实边长是 8cm x 8cm x 8cm
    )
    
    # 3. 将机械臂设置到一个比较容易观察的“举手”初始姿态
    q_init = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    sim.data.qpos[:7] = q_init
    
    # 强制更新一次运动学，让几何体刷新到正确的位置
    mujoco.mj_forward(sim.model, sim.data) 
    
    # ==========================================================
    # 静态观察模式 (只渲染，不计算物理)
    # ==========================================================
    print("\n" + "="*50)
    print("🛑 仿真物理时间已锁定（不会掉下来）！")
    print("👀 请在弹出的 MuJoCo 窗口中仔细检查蓝色负载块：")
    print("   1. 【位置】：是不是正好悬浮在夹爪的中央/前方？")
    print("   2. 【大小】：8cm 立方体的比例是否合适？")
    print("\n💡 提示：")
    print("   - 鼠标右键拖动：旋转视角")
    print("   - 鼠标滚轮：缩放视角")
    print("   - 鼠标左键双击负载块：可以聚焦视角")
    print("❌ 检查完毕后，在终端按 Ctrl+C 退出程序。")
    print("="*50 + "\n")
    
    try:
        # 死循环：只同步画面，绝对不调用 sim.step() 推进时间！
        while sim.is_alive():
            # 🔥 修改了这里：调用我们自己封装的 render，不仅刷新画面，还会把蓝块画上去！
            sim.render()
            time.sleep(0.02) # 50Hz 的刷新率，足够人眼观看且不占 CPU
    except KeyboardInterrupt:
        print("\n退出静态观察模式。")
    finally:
        sim.close()

if __name__ == "__main__":
    main()