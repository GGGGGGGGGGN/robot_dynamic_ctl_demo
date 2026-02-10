# realman_sim/assets/__init__.py
from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parent  # .../realman_sim/assets
REAL_MAN_MJCF = str(ASSETS_ROOT / "overseas_65_b_v_description/mjcf/realman_robot.xml")
REAL_MAN_MJCF_LEGACY = str(ASSETS_ROOT / "overseas_65_b_v_description/mjcf/realman_robot_legecy.xml")
REAL_MAN_URDF = str(ASSETS_ROOT / "overseas_65_b_v_description/urdf/model_for_mujoco.urdf")
REAL_MAN_STL = str(ASSETS_ROOT / "overseas_65_b_v_description/meshes")
def get_model_path():
    return REAL_MAN_MJCF_LEGACY