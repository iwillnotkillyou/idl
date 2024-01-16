"""SegmentationNN"""
import torch
import torch.nn as nn

class TransposeConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding= "valid"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        import torchvision
        self.backbone = torchvision.models.resnet18(True, weights=torchvision.models.ResNet18_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        #for param in self.backbone.layer2.parameters():
        #   param.requires_grad = True
        self.backbone.layer2 = nn.Identity()
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.sizes = self.hp["hidden_sizes"]
        self.tp_conv_sizes = self.hp["tp_conv_sizes"]
        self.ks = self.hp["ks"]
        embed0_channels = self.backbone.conv1.out_channels
        embed1_channels = [x for x in self.backbone.layer1[-1].children()][-2].out_channels
        self.conv0 = ConvLayer(embed1_channels, self.sizes[0], 3, 1, padding="same")
        self.convs = nn.Sequential(*[ConvLayer(self.sizes[i], self.sizes[i + 1], self.ks, 1)
                                     for i in range(len(self.sizes) - 1)])
        self.upconv1 = TransposeConvLayer(self.sizes[-1], self.tp_conv_sizes[0], 3, 2)
        self.conv1 = ConvLayer(self.tp_conv_sizes[0], self.tp_conv_sizes[0], 3, 1, padding="same")
        self.upconv2 = TransposeConvLayer(self.tp_conv_sizes[0] + embed0_channels, self.tp_conv_sizes[1], 3, 2)
        self.conv2 = ConvLayer(self.tp_conv_sizes[1] + 3, 23, 3, 1, padding="same")
        self.dp = nn.Dropout2d(p=0.2)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        original_size = x.shape[2:]
        bs = x.shape[0]
        inp = x
        x = torch.nn.functional.interpolate(x, size=(224, 224))
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        embeds0 = x
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)

        def f(x,embeds):
            s = x.shape[2:]
            embed_ss = nn.functional.interpolate(embeds, s)
            x = torch.concat([x, embed_ss], 1)
            return x


        x = self.conv0(x)
        x = self.dp(x)
        x = self.convs(x)
        x = nn.functional.interpolate(x, (60,60))
        x = self.upconv1(x)
        x = self.conv1(x)
        x = self.dp(x)
        x = self.upconv2(f(x,embeds0))
        if not (x.shape[2] > 180 and x.shape[2] < 260):
          print(f"shape sucks {x.shape[2]}")
        x = nn.functional.interpolate(x, original_size)
        x = self.conv2(f(x,inp))

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()


if __name__ == "__main__":
    from torchinfo import summary

    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")
