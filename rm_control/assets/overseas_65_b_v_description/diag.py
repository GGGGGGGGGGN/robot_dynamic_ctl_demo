import os
import mujoco
import mujoco.viewer
import re

# ================= 1. 自动定位关键路径 =================
# 获取当前脚本所在目录 (假设脚本在 urdf 文件夹里)
script_dir = os.path.dirname(os.path.abspath(__file__))

# 你的 URDF 文件
urdf_file_path = os.path.join(script_dir, "model_for_mujoco.urdf")

# 你的 Meshes 文件夹 (假设就在脚本旁边)
meshes_dir = os.path.join(script_dir, "meshes")

print(f"📍 脚本位置: {script_dir}")
print(f"📂 Meshes 文件夹位置: {meshes_dir}")

# ================= 2. 严厉的“体检” (Debug 关键) =================
# 我们先拿一个文件测试，看看 Python 能不能找到它
test_file = os.path.join(meshes_dir, "base_link_underpan.STL")
if not os.path.exists(test_file):
    print("\n❌ 致命错误：Python 找不到 Mesh 文件！")
    print(f"   Python 试图寻找: {test_file}")
    print("   👉 请检查：")
    print("      1. meshes 文件夹里真的有这个文件吗？")
    print("      2. 文件名大小写完全匹配吗？(比如 .stl 和 .STL)")
    exit(1) # 如果这步挂了，后面不用跑了，肯定是文件路径不对
else:
    print(f"✅ Python 成功找到了: {os.path.basename(test_file)}")

# ================= 3. 内存“偷天换日” =================
print("\n🔧 正在准备加载...")
with open(urdf_file_path, 'r', encoding='utf-8') as f:
    urdf_content = f.read()

# 这里的逻辑是：不管你 URDF 里写的是 meshes/ 还是 ../meshes/
# 只要我看到文件名，我就把它替换成【绝对路径】
def inject_absolute_path(match):
    # 获取文件名 (base_link_underpan.STL)
    filename = os.path.basename(match.group(1))
    # 拼接绝对路径
    abs_path = os.path.join(meshes_dir, filename)
    return f'filename="{abs_path}"'

# 正则匹配 filename="..."
pattern = re.compile(r'filename="([^"]+\.(?:STL|stl))"', re.IGNORECASE)
fixed_content = pattern.sub(inject_absolute_path, urdf_content)

# ================= 4. 喂给 MuJoCo =================
try:
    # 注意：这里用 from_xml_string，不走文件，直接走内存
    model = mujoco.MjModel.from_xml_string(fixed_content)
    data = mujoco.MjData(model)
    print("\n🎉🎉🎉 成功！MuJoCo 加载成功！")
    print("   (我们通过 Python 喂给了 MuJoCo 绝对路径，绕过了它的路径解析坑)")
    
    # 启动
    mujoco.viewer.launch(model, data)

except Exception as e:
    print(f"\n❌ MuJoCo 依然报错: {e}")