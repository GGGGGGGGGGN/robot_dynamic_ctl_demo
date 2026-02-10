
import os
import re
import mujoco

def main():
    # ================= 配置区域 =================
    # 获取当前脚本所在目录 (即 description 根目录)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义输入输出目录
    urdf_dir = os.path.join(root_dir, "urdf")
    meshes_dir = os.path.join(root_dir, "meshes")
    output_dir = os.path.join(root_dir, "mjcf")
    
    # 自动寻找 urdf 目录下得第一个 .urdf 文件
    urdf_files = [f for f in os.listdir(urdf_dir) if f.endswith('.urdf')]
    if not urdf_files:
        print(f"❌ 错误：在 {urdf_dir} 下没找到 .urdf 文件！")
        return
    
    input_urdf_path = os.path.join(urdf_dir, urdf_files[0])
    # 定义输出文件名
    output_urdf_name = "realman_mujoco.urdf"
    output_xml_name = "realman_mujoco.xml"
    
    output_urdf_path = os.path.join(output_dir, output_urdf_name)
    output_xml_path = os.path.join(output_dir, output_xml_name)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📂 创建目录: {output_dir}")

    print(f"📖 正在读取: {input_urdf_path}")
    with open(input_urdf_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # ================= 步骤 1: 暴力清洗路径 =================
    # 目标：把所有 filename=".../xxx.STL" 变成 filename="xxx.STL"
    # 这样配合 meshdir 就能完美工作
    print("🧹 正在清洗 STL 路径...")
    
    def strip_path(match):
        full_path = match.group(1)
        filename = os.path.basename(full_path) # 只保留文件名
        return f'filename="{filename}"'

    # 正则匹配 filename="..."
    pattern_mesh = re.compile(r'filename="([^"]+\.(?:STL|stl))"', re.IGNORECASE)
    content_clean = pattern_mesh.sub(strip_path, content)

    # ================= 步骤 2: 注入 MuJoCo 编译器配置 =================
    # 目标：在 <robot name="..."> 后面插入 <mujoco> 标签
    # meshdir="../meshes" 的意思是：从 mjcf 文件夹往上一级，再进 meshes
    print("💉 正在注入 MuJoCo <compiler> 配置...")
    
    mujoco_tag = """
  <mujoco>
    <compiler meshdir="../meshes" discardvisual="false" balanceinertia="true"/>
  </mujoco>
"""
    # 找到 <robot ...> 标签的结束位置
    # 简单的正则查找 <robot ...>
    pattern_robot = re.compile(r'(<robot[^>]*>)', re.IGNORECASE)
    
    if pattern_robot.search(content_clean):
        # 在 <robot ...> 后面插入 mujoco tag
        content_final = pattern_robot.sub(r'\1' + mujoco_tag, content_clean)
    else:
        print("⚠️ 警告：没找到 <robot> 标签，直接追加到开头（可能会出错）")
        content_final = mujoco_tag + content_clean

    # ================= 步骤 3: 保存修改后的 URDF =================
    with open(output_urdf_path, 'w', encoding='utf-8') as f:
        f.write(content_final)
    print(f"💾 已保存修正版 URDF: {output_urdf_path}")

    # ================= 步骤 4: 转换为 MJCF XML =================
    print("🚀 正在转换为原生 MJCF XML...")
    try:
        # 加载刚才生成的 URDF
        # 因为我们已经设置了 meshdir="../meshes"，MuJoCo 应该能找到文件
        model = mujoco.MjModel.from_xml_path(output_urdf_path)
        
        # 保存为 XML
        mujoco.mj_saveLastXML(output_xml_path, model)
        print(f"🎉 转换成功！XML 已保存: {output_xml_path}")
        print("-" * 30)
        print("👉 以后在 load_model.py 中，请加载这个文件：")
        print(f"   '{output_xml_path}'")
        print("-" * 30)

    except Exception as e:
        print(f"❌ 转换 XML 失败: {e}")
        print("提示：请检查 mjcf/realman_mujoco.urdf 里的 meshdir 是否正确指向了 meshes 文件夹")

if __name__ == "__main__":
    main()