# realman_sim/assets/__init__.py
from pathlib import Path

ASSETS_ROOT = Path(__file__).resolve().parent  # .../realman_sim/assets
PANDA_URDF= str(ASSETS_ROOT / "panda_description/urdf/panda.urdf")
PANDA_XML= str(ASSETS_ROOT / "franka_emika_panda/scene.xml")


def get_model_path_xml():
    return PANDA_XML

def get_model_path_urdf():
    return PANDA_URDF