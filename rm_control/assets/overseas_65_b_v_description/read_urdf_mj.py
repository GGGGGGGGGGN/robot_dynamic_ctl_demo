import os
import mujoco

# ================= 1. 路径设置 =================
# 获取脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

urdf_file = os.path.join(current_dir,"model_for_mujoco.urdf")

# ================= 2. 存在性检查 =================
if not os.path.exists(urdf_file):
    print(f"❌ 错误: 找不到输入文件 {urdf_file}")
    exit(1)

# ================= 6. 尝试加载 =================
try:
    print("🚀 尝试调用 MuJoCo 加载...")
    print(urdf_file)
    model = mujoco.MjModel.from_xml_path(urdf_file)
    print(f"✅ ✅ ✅ 成功！MuJoCo 成功加载了新文件！")
except Exception as e:
    print(f"❌ MuJoCo 加载失败: {e}")