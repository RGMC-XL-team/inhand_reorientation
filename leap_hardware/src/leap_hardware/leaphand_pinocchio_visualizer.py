import os

import numpy as np
import pinocchio
import rospkg
from pinocchio.visualize import MeshcatVisualizer


class LeapHandPinocchioVisualizer:
    def __init__(self, with_obj=False):
        self.with_obj = with_obj

    def _setup_visualizer(self):
        rospack = rospkg.RosPack()
        if self.with_obj:
            pkg_dir = rospack.get_path("leap_sim")
            urdf_model_path = os.path.join(pkg_dir, "assets/leap_hand", "leaphand_cube.urdf")
            mesh_dir = os.path.join(pkg_dir, "assets/leap_hand")
        else:
            pkg_dir = rospack.get_path("my_robot_description")
            urdf_model_path = os.path.join(pkg_dir, "urdf", "leaphand.urdf")
            mesh_dir = os.path.join(pkg_dir, "urdf/leap_hand/mesg")

        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(urdf_model_path, mesh_dir)
        self.viz = MeshcatVisualizer(model, collision_model, visual_model)

        # initialize viewer
        try:
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            return True
        except ImportError as err:
            print("Error while initializing the viewer. It seems you should install Python meshcat")
            print(err)
            return False

    def display(self, q0):
        if not isinstance(q0, np.ndarray):
            q0 = np.array(q0).flatten()
        if self.with_obj:
            assert len(q0) == 23
        else:
            assert len(q0) == 16
        self.viz.display(q0)
