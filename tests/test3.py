import mujoco
import numpy as np

# ⚠️ 请修改为你的 XML 实际路径
XML_PATH = "/Users/chenxu/Library/CloudStorage/OneDrive-Personal/Code/robot_dynamic_ctl_demo/rm_control/assets/franka_emika_panda/scene.xml"

def verify_torque_fidelity():
    print(f"🔬 启动力矩保真度验证...")
    print(f"📂 加载模型: {XML_PATH}")
    
    # 1. 加载模型 (Model)
    try:
        m = mujoco.MjModel.from_xml_path(XML_PATH)
    except ValueError:
        print("❌ 找不到文件，请检查路径！")
        return

    # ---------------------------------------------------------
    # 🔥 [关键步骤] 修改模型参数 (Modify Model)
    # 必须在创建 MjData 之前或之后修改，但某些参数修改后推荐重置 Data
    # ---------------------------------------------------------
    print("\n🛠️  正在执行‘纯力矩模式’强制转换...")
    
    target_torque = 20.0  # 我们想要测试的目标力矩
    
    for i in range(m.nu): # 遍历所有执行器
        # A. 强制类型转换 (最重要！防止 filter 或 affine 干扰)
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED  # 固定增益
        m.actuator_dyntype[i]  = mujoco.mjtDyn.mjDYN_NONE     # 无动力学延迟
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE   # 无偏置(无弹簧阻尼)
        
        # B. 数值设定
        m.actuator_gainprm[i, 0] = 1.0  # 增益 = 1.0
        m.actuator_biasprm[i, :] = 0.0  # 偏置 = 0.0
        
        # C. 🔓 [破案关键] 解除限幅！
        # 你的 XML 里 actuator2 限制了 [-1.76, 1.76]。不改这个，永远测不到 20。
        m.actuator_ctrlrange[i, :] = [-1000.0, 1000.0]
        m.actuator_forcerange[i, :] = [-1000.0, 1000.0]
        
    print("✅ 模型参数已修正：Gain=1, Bias=None, No Limits.")

    # ---------------------------------------------------------
    # 2. 创建数据 (Data)
    # ---------------------------------------------------------
    d = mujoco.MjData(m)
    
    # 3. 设置测试条件
    # 给所有关节输入 20 Nm
    d.ctrl[:7] = target_torque
    
    # 随便给个姿态，证明力矩与位置无关 (因为我们关掉了 affine gain)
    d.qpos[:7] = [0, -0.5, 0, -2, 0, 1.5, 0.7]

    # ---------------------------------------------------------
    # 🔥 [计算管线] 
    # 1. mj_fwdPosition: 更新几何信息 (力臂 Moment Arm)
    # 2. mj_fwdActuation: 计算电机输出
    # ---------------------------------------------------------
    mujoco.mj_fwdPosition(m, d)
    mujoco.mj_fwdActuation(m, d)
    
    # 4. 获取结果
    # qfrc_actuator 是经过传动后作用在关节上的最终力矩
    real_torque = d.qfrc_actuator[:7]
    
    # ---------------------------------------------------------
    # 📊 打印报告
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print(f"🎯 目标力矩: {target_torque} Nm")
    print("="*40)
    
    all_passed = True
    for i in range(7):
        act_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        out = real_torque[i]
        
        # 你的 actuator2 之前是 1.76，现在应该是 20
        status = "✅" if abs(out - target_torque) < 1e-4 else "❌"
        if "❌" in status: all_passed = False
            
        print(f"Joint {i+1} ({act_name}): {out:.4f} Nm  {status}")
        
    print("-" * 40)
    
    if all_passed:
        print("🎉 验证成功！输入等于输出，纯力矩模式已激活。")
        print("💡 结论：之前的 1.76 是被 XML 里的 ctrlrange 截断了。")
    else:
        print("💀 验证失败：请检查 XML 是否有 gear!=1 或者其他插件干扰。")

if __name__ == "__main__":
    verify_torque_fidelity()