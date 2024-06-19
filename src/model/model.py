import torch
import torch.nn as nn
from src.util.utils import map_to_UV

class InstantUV(nn.Module):
    """
    TODO write docstring
    Model that learns the stored texture in the image implicitly in 2D using Instant-NGP

    During training the input to our model will be our preprocessed points 
    i.e. 2D points from an image which represents a UV map.

    The output should be an 3 channel (normalized?) vector representing the RGB values. 
    For understanding check the dummy model below.
    """
    
    def __init__(self):
        # TODO
        self.mapping = None # TODO
        pass

    def forward(self, points_uv):
        # TODO
        pass

    @torch.no_grad()
    def inference(self, points_xyz):
        self.eval()
        points_uv = map_to_UV(points_xyz)
        return self.forward(points_uv)


class DummyInstantUV(nn.Module):
    """
    Dummy model -- only for understanding
    """
    
    def __init__(self, H, W):
        # TODO
        self.mapping = None # TODO
        self.model = nn.Sequential(
            nn.Linear(H*W, 3), # map 2d points to RGB values
            nn.Softmax() # squish rgb values between 0 and 1
        )

    def forward(self, points_uv):
        return self.model(points_uv)

    @torch.no_grad()
    def inference(self, point_xyz):
        self.eval()
        point_uv = map_to_UV(point_xyz)
        return self.forward(point_uv)
