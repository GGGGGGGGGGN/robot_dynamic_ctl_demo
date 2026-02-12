import numpy as np
from rm_control.simulation.sim_interface import SimInterface
from rm_control.dynamics.pinocchio_dynamics import PinocchioDynamics
from rm_control.assets import get_model_path_xml, get_model_path_urdf
import mujoco

def debug():
    print("----- Debugging Joint Loading Mismatch -----")
    
    # 1. SimInterface (MuJoCo) with active_joint_names=None
    print("\n[MuJoCo] Loading with active_joint_names=None...")
    sim = SimInterface(get_model_path_xml(), active_joint_names=None, render=False)
    
    mj_joint_names = []
    for jid in sim.active_jnt_ids:
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        mj_joint_names.append(name)
        
    print(f"Num Active Joints: {len(mj_joint_names)}")
    print(f"Joint Names: {mj_joint_names}")
    
    # 2. Pinocchio with active_joint_names=None
    print("\n[Pinocchio] Loading with active_joint_names=None...")
    pin_dyn = PinocchioDynamics(get_model_path_urdf(), active_joint_names=None, ee_name="panda_link7")
    
    pin_joint_names = pin_dyn.model.names.tolist() # Includes 'universe' usually
    # Filter out universe and fixed joints? Pinocchio model.names includes *all* joints in reduced model.
    # But if active_joint_names=None, it uses full model.
    # full_model usually includes fixed joints too? No, pinocchio model only has mobile joints in .nq/.nv, but names list has all?
    # Actually pinocchio model.names corresponds to joints 0..njoints.
    
    print(f"Num Joints (nq={pin_dyn.model.nq}, nv={pin_dyn.model.nv})")
    print(f"Joint Names: {pin_joint_names}")
    
    # Check for mismatch
    if len(mj_joint_names) != pin_dyn.model.nv:
        print("\n❌ MISMATCH DETECTED: Number of DOFs differ!")
        print(f"MuJoCo: {len(mj_joint_names)}")
        print(f"Pinocchio: {pin_dyn.model.nv}")
    else:
        print("\n✅ Number of DOFs match.")
        
    # Check name match (ignoring prefix if needed)
    # MuJoCo names: "joint1", "joint2"...
    # URDF names: "panda_joint1", "panda_joint2"...
    
    print("\nComparing names...")
    for i in range(min(len(mj_joint_names), pin_dyn.model.nv)):
        mj_name = mj_joint_names[i]
        # Pinocchio joint 0 is universe, so we look at i+1?
        # pin.model.names[0] is universe.
        # pin.model.names[1] is first joint.
        pin_name = pin_joint_names[i+1] 
        print(f"  {i}: MuJoCo='{mj_name}' vs Pinocchio='{pin_name}'")

if __name__ == "__main__":
    debug()
