from torch import nn
from transformers.image_utils import ChannelDimension
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DaEncoder(nn.Module):
    def __init__(self):
        super(DaEncoder, self).__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("depth-anything/prompt-depth-anything-vits-hf", use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/prompt-depth-anything-vits-hf")

    def forward(self, image):
        b = image.shape[0]
        hw = image.shape[-2:]
        call_kwargs = {"do_rescale": False}
        inputs = self.image_processor(
            images=image,
            return_tensors="pt",
            data_format=ChannelDimension.FIRST,
            **call_kwargs,
        )
        outputs = self.model(**inputs)
        return outputs.predicted_depth
