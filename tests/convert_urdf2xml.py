import mujoco
import os

# 1. 定义路径
# 请确保你的 urdf 文件在这个路径下
# 注意：你需要把 meshes 文件夹放在和 urdf 同一级，或者修改 urdf 里的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设你的 urdf 在 rm_control/assets/ 下
urdf_path = os.path.join(current_dir, "../rm_control/assets/overseas_65_b_v_description/urdf/model_for_mujoco.urdf")
save_path = os.path.join(current_dir, "../rm_control/assets/overseas_65_b_v_description/mjcf/model_for_mujoco.xml")

print(f"正在加载 URDF: {urdf_path}")

try:
    # 2. 让 MuJoCo 加载 URDF
    # 这一步会自动计算所有的 position 和 quaternion
    model = mujoco.MjModel.from_xml_path(urdf_path)
    
    # 3. 保存为 XML
    mujoco.mj_saveLastXML(save_path, model)
    print(f"✅ 转换成功！已保存为: {save_path}")
    print("这个文件的坐标和旋转绝对是正确的。")

except Exception as e:
    print(f"❌ 转换失败: {e}")
    print("提示：请检查 urdf 文件里的 mesh 路径是否正确")