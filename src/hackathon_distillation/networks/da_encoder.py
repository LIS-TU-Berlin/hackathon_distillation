from torch import nn, Tensor
from transformers.image_utils import ChannelDimension
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DaEncoder(nn.Module):
    """DepthAnything Encoder"""
    def __init__(self, dmin: float = 0., dmax: float = 5.0):
        super(DaEncoder, self).__init__()
        model = "depth-anything/Depth-Anything-V2-Small-hf"
        #model = "depth-anything/prompt-depth-anything-vits-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model)
        self.dmin = dmin
        self.dmax = dmax

    def forward(self, image: Tensor, postprocess: bool = True, depth: Tensor = None) -> Tensor:
        call_kwargs = {}
        #call_kwargs = {"do_rescale": False}
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            data_format=ChannelDimension.FIRST,
            #prompt_depth=depth,
            **call_kwargs,
        )
        outputs = self.model(**inputs)
        if not postprocess:
            pred_post = outputs.predicted_depth
        else:
            pred_post = self.image_processor.post_process_depth_estimation(outputs)[0]["predicted_depth"][None]
            pred_post = pred_post * 0.5 + 0.2 #(1. - 0.1) + 0.1
            #pred_post = (pred_post - self.dmin) / (self.dmax - self.dmin + 1e-8)

        return 1 / pred_post

