import torch
import torch.nn as nn
import torchvision.models as models


class DepthImageEncoder(nn.Module):
    """
    ResNet-based encoder for depth images.
    Input: (B, 1, H, W)
    Output: (B, feature_dim)
    """
    def __init__(self, feature_dim=256, pretrained=False, freeze_layers=False):
        super(DepthImageEncoder, self).__init__()
        self.feature_dim = feature_dim

        # Start with a ResNet18 (or use resnet34, etc.)
        base_model = models.resnet18(pretrained=pretrained)

        # Modify first conv layer to accept 1 channel (depth image)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = base_model.conv1.weight.data.sum(dim=1, keepdim=True) / 3.0  # average RGB weights

        # Keep rest of the layers
        self.encoder = nn.Sequential(
            self.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,  # Output shape: (B, 512, 1, 1)
        )

        # Freeze later layers to reduce overfitting
        if freeze_layers:
            # Freeze layer3 and layer4 (the deepest, most specific layers)
            for param in self.encoder[6].parameters():  # layer3
                param.requires_grad = False
            for param in self.encoder[7].parameters():  # layer4
                param.requires_grad = False
            print("Frozen ResNet layer3 and layer4")

        # Final projection layer
        self.fc = nn.Linear(512, feature_dim)

        # # Final projection layer with dropout
        # self.fc = nn.Sequential(
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(512, feature_dim)
        # )

    def forward(self, x):
        """
        Args:
            x: Depth image tensor of shape (B, 1, H, W)
        Returns:
            Feature vector of shape (B, feature_dim)
        """
        x = self.encoder(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.fc(x)  # (B, feature_dim)
        return x


if __name__ == "__main__":
    import einops
    import torch as th


    encoder = DepthImageEncoder()

    imgs = th.randn(64, 2, 1, 360, 640)
    imgs = einops.rearrange(imgs, "b s ... -> (b s) ...")
    print(imgs.shape)
    x = encoder(imgs)
    x_out = einops.rearrange(x, "(b s) ... -> b s (...)", b=64, s=2)

    print(x.shape, x_out.shape)
