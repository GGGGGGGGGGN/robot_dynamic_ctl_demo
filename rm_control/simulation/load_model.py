import time
import numpy as np
import mujoco
import mujoco.viewer

# 导入我们在 rm_control/__init__.py 里写好的路径工具
# 如果报错 "ModuleNotFoundError"，请确保你已经 pip install -e . 安装了包
from rm_control.assets import get_model_path

def main():
    # 1. 加载模型
    # 使用我们刚写好的 全身动力学模型
    xml_path = get_model_path()
    print(f"Loading model from: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 打印动力学信息 (导师重点检查项)
    print("-" * 30)
    print(f"机器人状态维度 (nq - qpos): {model.nq}")
    print(f"机器人速度维度 (nv - qvel): {model.nv}")
    print(f"控制输入维度   (nu - ctrl): {model.nu}")
    print("-" * 30)

    # 简单验证：如果 nu 不等于 15 (1腰+2头+12臂)，说明模型定义有误
    expected_nu = 15
    if model.nu != expected_nu:
        print(f"⚠️ 警告: 执行器数量 ({model.nu}) 与预期 ({expected_nu}) 不符，请检查 XML！")
    else:
        print("✅ 模型执行器配置正确，准备起飞！")

    # 3. 启动仿真查看器 (Passive 模式，允许我们自己控制循环)
    # launch_passive 允许我们在 while 循环里插入控制代码
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # 初始化一下机器人姿态，别让它躺在地上
        # 稍微把手臂抬起来一点 (这里只是简单的硬编码，以后会用控制器)
        # data.qpos[7] 是轮子或者腰的位置，具体要看 xml 定义顺序
        # 这里先不做操作，让它受重力自然下落看看物理对不对
        
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()

            # --- 控制算法区域 (Controller Area) ---
            # 目前给全 0 力矩，机器人应该会因为重力垮掉
            # data.ctrl[:] = 0.0 
            # ------------------------------------

            # 物理步进 (Step Physics)
            mujoco.mj_step(model, data)

            # 更新画面 (Render)
            viewer.sync()

            # 保持实时性 (Time keeping)
            # MuJoCo 默认 timestep 是 0.001 (1ms)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()