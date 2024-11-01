from torch import nn

class CustomViT(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional encoder
        self.conv_encoder = nn.Sequential(
                                    nn.Conv2d(1, 16, 3, 1, 1),
                                    nn.InstanceNorm2d(16),
                                    nn.GELU(),
                                    nn.Conv2d(16, 32, 3, 1, 1),
                                    nn.InstanceNorm2d(32),
                                    nn.GELU(),
                                    nn.Conv2d(32, 64, 3, 1, 1),
                                    nn.InstanceNorm2d(64),
                                    nn.GELU(),
                                    nn.Conv2d(64, 32, 3, 1, 1),
                                    nn.InstanceNorm2d(32),
                                    nn.GELU(),
                                    nn.Conv2d(32, 16, 3, 1, 1),
                                    nn.InstanceNorm2d(16),
                                    nn.GELU(),
                                    nn.Conv2d(16, 1, 3, 1, 1),
                                    nn.InstanceNorm2d(1),
                                    nn.GELU())

    def forward(self, x):
        x = self.conv_encoder(x)
        print(x.shape)
        return x