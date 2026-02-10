import os
import mujoco

# ================= 1. 路径设置 =================
# 获取脚本所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接出文件的完整绝对路径
input_file = os.path.join(script_dir, "overseas_65_b_v_description.urdf")
output_file = os.path.join(script_dir, "model_for_mujoco.urdf")

# ================= 2. 存在性检查 =================
if not os.path.exists(input_file):
    print(f"❌ 错误: 找不到输入文件 {input_file}")
    exit(1)

# ================= 3. 读取与替换 =================
print(f"📖 正在读取: {os.path.basename(input_file)}")
with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# 定义我们要查找和替换的目标
target_str = "package://overseas_65_b_v_description/"
replace_str = "" 

# ★ 关键检查 1：源文件里到底有没有这个字符串？
count = content.count(target_str)
if count == 0:
    print(f"⚠️  警告！在源文件中未找到字符串: '{target_str}'")
    print("    -> 这意味着 replace 操作不会起任何作用！")
    print("    -> 请检查 URDF 文件中的 package 名称是否完全一致（空格、大小写）。")
    
    # 打印一行原始包含 stl 的内容来看看它长什么样
    for line in content.split('\n'):
        if ".STL" in line or ".stl" in line:
            print(f"    [源文件样本]: {line.strip()}")
            break
else:
    print(f"✅ 在源文件中找到 {count} 处匹配，准备替换...")

# 执行替换
new_content = content.replace(target_str, replace_str)

# ================= 4. 写入与回读验证 =================
print(f"💾 正在写入: {os.path.basename(output_file)}")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(new_content)

# ★ 关键检查 2：读取刚才写入的文件，看看它是啥样
print("-" * 30)
print("🧐 [核查生成的文件内容]")
with open(output_file, "r", encoding="utf-8") as f:
    check_lines = f.readlines()

found_mesh_line = False
first_mesh_path = ""

for i, line in enumerate(check_lines):
    if ".STL" in line or ".stl" in line:
        print(f"    行 {i+1}: {line.strip()}")
        found_mesh_line = True
        # 提取引号里的路径来做最后一步验证
        # 假设格式是 filename="路径"
        if 'filename="' in line:
            parts = line.split('filename="')
            if len(parts) > 1:
                first_mesh_path = parts[1].split('"')[0]
        break

if not found_mesh_line:
    print("❌ 奇怪：在新文件中没找到任何 .STL 引用？")

print("-" * 30)

# ================= 5. 物理路径验证 =================
# 如果我们提取到了路径，我们帮 MuJoCo 跑一下腿，看看路径对不对
if first_mesh_path:
    # 模拟 MuJoCo 的解析逻辑：相对于 urdf 文件路径解析
    # script_dir 就是 urdf 所在的目录
    resolved_path = os.path.abspath(os.path.join(script_dir, first_mesh_path))
    print(f"🕵️ [路径侦探]")
    print(f"    URDF写的是: {first_mesh_path}")
    print(f"    推算绝对路径: {resolved_path}")
    
    if os.path.exists(resolved_path):
        print(f"    ✅ 文件系统检查: 文件存在！MuJoCo 应该能读取。")
    else:
        print(f"    ❌ 文件系统检查: 文件不存在！")
        print(f"    👉 请检查 ../ 是否真的指向了 meshes 文件夹。")
print("-" * 30)

# ================= 6. 尝试加载 =================
try:
    print("🚀 尝试调用 MuJoCo 加载...")
    print(output_file)
    model = mujoco.MjModel.from_xml_path(output_file)
    print(f"✅ ✅ ✅ 成功！MuJoCo 成功加载了新文件！")
except Exception as e:
    print(f"❌ MuJoCo 加载失败: {e}")