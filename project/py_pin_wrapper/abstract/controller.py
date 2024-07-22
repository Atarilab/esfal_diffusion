import numpy as np

from bullet_utils.wrapper import PinBulletWrapper

class ControllerAbstract(object):
    def __init__(self,
                 robot: PinBulletWrapper,
                 **kwargs,
                 ) -> None:
        self.robot = robot
        self.diverged = False
        
    def get_torques(self,
                    q:np.array,
                    v:np.array,
                    **kwargs,
                    ) -> dict[float] :
        return {}