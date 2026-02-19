import mujoco
import numpy as np

class MujocoDynamics:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nv = model.nv
        self.M = np.zeros((self.nv, self.nv))

    def get_dynamics(self):
        """返回 M, h (h = C*dq + g)"""
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        h = self.data.qfrc_bias.copy()
        return self.M, h

    def get_jacobian(self, body_name):
        """返回位置雅可比 (3, nv)"""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv)) # 占位
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bid)
        return jacp