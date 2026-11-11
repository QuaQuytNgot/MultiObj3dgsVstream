"""CPU camera controls. Calibration, including principal point, survives motion."""
import math
import numpy as np
from ...shared.geometry import Camera


class CameraController:
    def __init__(self, width=1024, height=768, target=(0,0,0), radius=3.8):
        self.target = np.asarray(target, dtype=float)
        self.radius = float(radius)
        self.yaw, self.pitch = 0.0, 0.0
        self.free = False
        self.width, self.height = width, height
        f = .9*min(width,height)
        self.K = np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
        self._camera = None
        self.changed = True

    def adopt(self, camera, target=None):
        self.width,self.height = camera.width,camera.height
        self.K = camera.K.copy()
        if target is not None:
            self.target = np.asarray(target,dtype=float)
        delta = camera.center-self.target
        self.radius = max(.01,float(np.linalg.norm(delta)))
        self.yaw = math.atan2(delta[0],delta[2])
        self.pitch = math.asin(np.clip(delta[1]/self.radius,-1,1))
        self._camera = camera
        self.changed = False

    def fit(self, centers, scales=None):
        centers = np.asarray(centers,dtype=float)
        if centers.size == 0:
            return
        self.target = centers.mean(axis=0)
        extent = float(np.linalg.norm(centers-self.target,axis=1).max())
        self.radius = max(self.radius, extent+3.8*max(scales or [1.]))
        self._invalidate()

    def _invalidate(self):
        self._camera = None
        self.changed = True

    def orbit(self, dx, dy):
        self.yaw -= dx*.005
        self.pitch = float(np.clip(self.pitch+dy*.005,-1.5,1.5))
        self._invalidate()

    def zoom(self, delta):
        self.radius = float(np.clip(self.radius*math.exp(-delta*.1),.01,1e4))
        self._invalidate()

    def pan(self, dx, dy):
        camera = self.camera()
        right,down = camera.world_to_camera[0,:3],camera.world_to_camera[1,:3]
        self.target += (-dx*right-dy*down)*(self.radius/max(1,self.height))
        self._invalidate()

    def move(self, right, down, forward, dt):
        camera = self.camera()
        direction = camera.world_to_camera[:3,:3].T @ np.array([right,down,forward])
        self.target += direction*dt*max(.5,self.radius)
        self._invalidate()

    def resize(self,width,height):
        if width <= 0 or height <= 0:
            return
        old = self.camera()
        resized = old.resized(width,height)
        self.width,self.height,self.K = width,height,resized.K.copy()
        self._camera = resized
        self.changed = True

    def camera(self):
        if self._camera is None:
            offset = self.radius*np.array([math.sin(self.yaw)*math.cos(self.pitch),
                                          math.sin(self.pitch),math.cos(self.yaw)*math.cos(self.pitch)])
            eye = self.target+offset
            forward = -offset/np.linalg.norm(offset)
            right = np.cross([0.,-1.,0.],forward)
            right /= np.linalg.norm(right)
            down = np.cross(forward,right)
            rotation = np.stack([right,down,forward])
            w2c = np.eye(4)
            w2c[:3,:3],w2c[:3,3] = rotation,-rotation@eye
            self._camera = Camera(self.width,self.height,w2c,self.K)
        return self._camera
