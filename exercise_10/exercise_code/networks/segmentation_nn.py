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
        embed1_channels = [x for x in self.backbone.layer1[-1].children()][-1].num_features
        #self.conv00 = ConvLayer(embed0_channels, self.sizes[0]//3, 3, 2)
        self.conv01 = ConvLayer(embed0_channels, self.sizes[0] // 2, 3, 2)
        self.transform_dim1 = 32
        self.conv02 = ConvLayer(embed1_channels, self.sizes[0] // 2 + self.sizes[0] % 2, 3, 1, padding="same")
        self.convs = nn.Sequential(*[ConvLayer(self.sizes[i], self.sizes[i + 1], self.ks, 1, padding="same")
                                     for i in range(len(self.sizes) - 1)])
        self.transform_conv1 = ConvLayer(embed1_channels, self.transform_dim1*self.transform_dim1, 2, 1)
        self.upconv1 = TransposeConvLayer(self.sizes[-1], self.tp_conv_sizes[0], 3, 2)
        self.conv1 = ConvLayer(self.tp_conv_sizes[0], self.tp_conv_sizes[0], 3, 1, padding="same")
        self.upconv2 = TransposeConvLayer(self.tp_conv_sizes[0] + embed0_channels, self.tp_conv_sizes[1], 3, 2)
        self.conv2 = ConvLayer(self.tp_conv_sizes[1] + 3, 23, 3, 1, padding="same")
        self.register_buffer('identity1', torch.diag(torch.ones(self.transform_dim1)))

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
        #dentity1 = self.identity1.repeat(bs, 1, 1)
        #transform1 = (identity1 + torch.max(self.transform_conv1(x).view(bs, self.transform_dim1 * self.transform_dim1, -1), 2)[0].view(bs, self.transform_dim1, self.transform_dim1))

        def f(x,embeds):
            s = x.shape[2:]
            embed_ss = nn.functional.interpolate(embeds, s)
            x = torch.concat([x, embed_ss], 1)
            return x

        def f2(x, transform):
            s = x.shape[2:]
            td = transform.shape[1]
            v = (torch.bmm(x.view(bs,td,-1).transpose(1,2), transform).transpose(1,2)).view(bs,td,*s)
            return torch.concat([x,v],1)

        def f0(x,embeds0):
            b = nn.functional.interpolate(self.conv01(embeds0), (56, 56))
            c = self.conv02(x)
            return torch.concatenate([b, c], 1)


        x = f0(x,embeds0)
        x = self.convs(x)
        x = self.upconv1(x)
        x = self.conv1(x)
        x = self.upconv2(f(x,embeds0))
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
