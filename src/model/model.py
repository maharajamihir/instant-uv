import sys

import os
import sys
from pathlib import Path

# Append src/
sys.path.append(str(Path(__file__).parent.parent))
# Change into instant-uv
os.chdir(Path(__file__).parent.parent.parent)
# Append scripts DIR
SCRIPTS_DIR = str(Path(__file__).parent.parent / "tiny-cuda-nn/scripts")
sys.path.append(SCRIPTS_DIR)

import torch
import torch.nn as nn
from util.utils import map_to_UV

try:
    import tinycudann as tcnn

    print("Tiny CUDA nn successfully imported!")
except ImportError:
    print("This sample requires the tiny-cuda-nn extension for PyTorch.")
    print("You can install it by running:")
    print("============================================================")
    print("tiny-cuda-nn$ cd bindings/torch")
    print("tiny-cuda-nn/bindings/torch$ python setup.py install")
    print("============================================================")
    sys.exit()


class InstantUV(nn.Module):
    """
    TODO write docstring
    Model that learns the stored texture in the image implicitly in 2D using Instant-NGP

    During training the input to our model will be our preprocessed points 
    i.e. 2D points from an image which represents a UV map.

    The output should be an 3 channel (normalized?) vector representing the RGB values. 
    For understanding check the dummy model below.
    """

    def __init__(self, config):
        super().__init__()
        self.mapping = None  # TODO
        self.tiny_nn_config = {
            "n_input_dims": config["model"]["n_input_dims"],
            "n_output_dims": config["model"]["n_output_dims"],
            "encoding": config["model"]["encoding"],
            "network": config["model"]["network"]
        }

        model = tcnn.NetworkWithInputEncoding(n_input_dims=self.tiny_nn_config["n_input_dims"],
                                              n_output_dims=self.tiny_nn_config["n_output_dims"],
                                              encoding_config=self.tiny_nn_config["encoding"],
                                              network_config=self.tiny_nn_config["network"])
        self.model = model
        self.params = model.params

    def forward(self, points_uv):
        return self.model(points_uv)

    @torch.no_grad()
    def inference(self, points_xyz):
        raise NotImplementedError
        points_uv = map_to_UV(points_xyz)
        return self.forward(points_uv)


class InstantUV_VD1(nn.Module):
    """
    TODO write docstring
    Model that learns the stored texture in the image implicitly in 2D using Instant-NGP

    During training the input to our model will be our preprocessed points
    i.e. 2D points from an image which represents a UV map.

    The output should be an 3 channel (normalized?) vector representing the RGB values.
    For understanding check the dummy model below.
    """

    def __init__(self, tiny_nn_config):
        super().__init__()
        self.mapping = None  # TODO
        model = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=3,
                                              encoding_config=tiny_nn_config["encoding"],
                                              network_config=tiny_nn_config["network"])
        self.model = model
        self.params = model.params

    def forward(self, data):
        return self.model(data)


class DummyInstantUV(nn.Module):
    """
    Dummy model -- only for understanding
    """

    def __init__(self, H, W):
        # TODO
        self.mapping = None  # TODO
        self.model = nn.Sequential(
            nn.Linear(H * W, 3),  # map 2d points to RGB values
            nn.Softmax()  # squish rgb values between 0 and 1
        )

    def forward(self, points_uv):
        return self.model(points_uv)

    @torch.no_grad()
    def inference(self, point_xyz):
        raise NotImplementedError
        self.eval()
        point_uv = map_to_UV(point_xyz)
        return self.forward(point_uv)
