from torch import nn, Tensor
from transformers.image_utils import ChannelDimension
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DaEncoder(nn.Module):
    """DepthAnything Encoder"""
    def __init__(self, dmin: float = 0., dmax: float = 5.0):
        super(DaEncoder, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf", use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.dmin = dmin
        self.dmax = dmax

    def forward(self, image: Tensor, postprocess: bool = False) -> Tensor:
        call_kwargs = {}
        #call_kwargs = {"do_rescale": False}
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            data_format=ChannelDimension.FIRST,
            **call_kwargs,
        )
        outputs = self.model(**inputs)
        if not postprocess:
            return outputs.predicted_depth

        pred_post = self.image_processor.post_process_depth_estimation(outputs)[0]["predicted_depth"]
        depth = (pred_post - self.dmin) / (self.dmax - self.dmin + 1e-8)

        return depth
