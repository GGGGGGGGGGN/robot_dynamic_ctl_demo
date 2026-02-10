# realman_sim/assets/__init__.py
from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parent  # .../realman_sim/assets
REAL_MAN_MJCF_POSITION = str(ASSETS_ROOT / "overseas_65_b_v_description/mjcf/scene_pos.xml")
REAL_MAN_MJCF_TORQUE = str(ASSETS_ROOT / "overseas_65_b_v_description/mjcf/scene_torque.xml")

def get_model_path_position():
    return REAL_MAN_MJCF_POSITION

def get_model_path_torque():
    return REAL_MAN_MJCF_TORQUE