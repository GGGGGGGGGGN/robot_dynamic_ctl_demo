import time
import mujoco
import mujoco.viewer

from rm_control.assets import get_model_path
# 1. 加载模型
file_path = get_model_path()
model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

print("🚀 仿真开始！正在监听碰撞...")

# 2. 启动 Viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 物理步进
        mujoco.mj_step(model, data)
        
        # === 🕵️‍♂️ 碰撞侦探代码 ===
        # data.ncon 是当前接触点的数量
        if data.ncon > 0:
            print(f"⚠️ 检测到 {data.ncon} 个碰撞:")
            for i in range(data.ncon):
                contact = data.contact[i]
                
                # 获取碰撞几何体 ID
                geom1_id = contact.geom1
                geom2_id = contact.geom2
                
                # 获取所属 Body ID
                body1_id = model.geom_bodyid[geom1_id]
                body2_id = model.geom_bodyid[geom2_id]
                
                # 获取 Body 名字 (这就是你要找的元凶！)
                name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
                
                # 过滤掉地面的碰撞（如果你只关心机器人内部打架）
                if name1 != "base_link_underpan" and name2 != "base_link_underpan":
                     print(f"   💥 [打架现场] {name1}  <--->  {name2}")
            
            print("-" * 30)
            # 稍微暂停一下，不然刷屏太快看不清
            time.sleep(0.5) 
        # =========================

        viewer.sync()