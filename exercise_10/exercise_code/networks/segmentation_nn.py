"""SegmentationNN"""
import torch
import torch.nn as nn


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
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
        # for param in self.backbone.layer2.parameters():
        #    param.requires_grad = True
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.sizes = self.hp["hidden_sizes"]
        self.hp["use_transform1"] = True if not "use_transform1" in self.hp else self.hp["use_transform1"]
        self.hp["use_transform2"] = False if not "use_transform2" in self.hp else self.hp["use_transform2"]
        embed_channels = [x for x in self.backbone.layer1[-1].children()][-1].num_features
        embed2_channels = [x for x in self.backbone.layer2[-1].children()][-1].num_features
        self.conv0 = nn.Conv2d(embed2_channels, self.sizes[0], 1)
        self.convs = nn.Sequential(*[nn.Conv2d(self.sizes[i], self.sizes[i + 1], 3, 2 if i == 0 else 1, padding = "valid" if i == 0 else "same")
                                     for i in range(len(self.sizes) - 1)])
        self.transform_dim1 = 32
        self.transform_dim2 = 23
        self.transform_conv1 = nn.Conv2d(embed2_channels, self.transform_dim1*self.transform_dim1, 3, 2)
        if self.hp["use_transform2"]:
            self.transform_conv2 = nn.Conv2d(embed2_channels, self.transform_dim2*self.transform_dim2, 3, 2)
        self.upconv1 = nn.ConvTranspose2d(self.sizes[-1], 256, 3, 2)
        self.upconv2 = nn.ConvTranspose2d(256 + embed2_channels, 64, 3, 2)
        self.upconv3 = nn.ConvTranspose2d(64 + embed_channels, 32, 3, 2)
        self.upconv4 = nn.ConvTranspose2d(32 + 3, 23, 3, 2)
        self.register_buffer('identity1', torch.diag(torch.ones(self.transform_dim1)))
        self.register_buffer('identity2', torch.diag(torch.ones(self.transform_dim2)))

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
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        embeds1 = x
        x = self.backbone.layer2(x)
        embeds2 = x

        def f(x,embeds):
            s = x.shape[2:]
            embed_ss = nn.functional.interpolate(embeds, s)
            x = torch.concat([x, embed_ss], 1)
            return x

        identity1 = self.identity1.repeat(bs, 1, 1)
        identity2 = self.identity2.repeat(bs, 1, 1)
        if self.hp["use_transform1"]:
            transform1 = (identity1+torch.max(self.transform_conv1(x).view(bs,self.transform_dim1*self.transform_dim1,-1), 2)[0]
                      .view(bs, self.transform_dim1, self.transform_dim1))
        if self.hp["use_transform2"]:
            transform2 = (identity2+torch.max(self.transform_conv2(x).view(bs,self.transform_dim2*self.transform_dim2,-1), 2)[0]
                      .view(bs, self.transform_dim2, self.transform_dim2))

        x = self.conv0(x)


        def f2(x, transform):
            s = x.shape[2:]
            td = transform.shape[1]
            v = (torch.bmm(x.view(bs,td,-1).transpose(1,2), transform).transpose(1,2)).view(bs,td,*s)
            return torch.maximum(x,v)

        x = self.convs(x)
        x = self.upconv1(x)
        x = self.upconv2(f(x,embeds2))
        x = self.upconv3(f(x,embeds1))
        if self.hp["use_transform1"]:
            x = f2(x,transform1)
        x = self.upconv4(f(x,inp))
        if x.shape[2] < 190:
            raise Exception(f"too many layers {x.shape}")
        x = nn.functional.interpolate(x, original_size)
        if self.hp["use_transform2"]:
            x = f2(x,transform2)

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


class SegmentationNNOriginal(nn.Module):

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
        # for param in self.backbone.layer2.parameters():
        #    param.requires_grad = True
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        from torchvision import transforms
        self.sizes = self.hp["hidden_sizes"]
        if "addembeds" not in hp:
            hp["addembeds"] = True
        embed_channels = [x for x in self.backbone.layer1[-1].children()][-1].num_features
        embed2_channels = [x for x in self.backbone.layer2[-1].children()][-1].num_features
        extra_channels = embed_channels if hp["addembeds"] else 0
        ls = [nn.Conv2d(self.sizes[i] + extra_channels, self.sizes[i + 1], 2, 1 if i == 1 else 2) for i in
              range(len(self.sizes) - 1)]
        self.convs = nn.Sequential(nn.Conv2d(embed2_channels, self.sizes[0], 1)
                                   , *ls)
        self.upconv1 = nn.ConvTranspose2d(self.sizes[-1], 128, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.upconv4 = nn.ConvTranspose2d(64 + embed_channels, 16, 2, 2)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.last_conv = nn.Conv2d(16, 23, 1)

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
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        embeds = x
        x = self.backbone.layer2(x)

        def f(x):
            s = x.shape[2:]
            embed_ss = nn.functional.interpolate(embeds, s)
            x = torch.concat([x, embed_ss], 1)
            return x

        if self.hp["addembeds"]:
            m = len(self.convs)
            x = self.convs[0](x)

            for i in range(1, m):
                x = self.convs[i](f(x))

            x = self.upconv1(x)
            x = self.upconv2(x)
            x = self.upconv3(x)
            x = self.upconv4(f(x))
        else:
            x = self.convs(x)
            x = self.upconv1(x)
            x = self.upconv2(x)
            x = self.upconv3(x)
            x = self.upconv4(f(x))
        x = nn.functional.interpolate(x, original_size)
        x = self.last_conv(x)

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


class SegmentationNNUnet(torch.nn.Module):
    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        import torchvision
        self.backbone = torchvision.models.resnet18(True, weights=torchvision.models.ResNet18_Weights.DEFAULT)
        for param in self.backbone.parameters():
            param.requires_grad = False
        # for param in self.backbone.layer2.parameters():
        #    param.requires_grad = True
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.hp = hp
        self.cnns_down = torch.nn.ModuleList(list(torch.nn.ModuleList([None, None]) for x in range(3)))
        embed2_channels = [x for x in self.backbone.layer1[-1].children()][-1].num_features
        base = self.hp["base"]
        self.cnns_down[0][0] = torch.nn.Conv2d(embed2_channels, base, kernel_size=3)
        self.cnns_down[0][1] = torch.nn.Conv2d(base, base, kernel_size=3)
        self.cnns_down[1][0] = torch.nn.Conv2d(base, base * 2, kernel_size=3)
        self.cnns_down[1][1] = torch.nn.Conv2d(base * 2, base * 2, kernel_size=3)
        self.cnns_down[2][0] = torch.nn.Conv2d(base * 2, base * 4, kernel_size=3)
        self.cnns_down[2][1] = torch.nn.Conv2d(base * 4, base * 4, kernel_size=3)
        self.batch_norms_down = torch.nn.ModuleList(
            list(torch.nn.BatchNorm2d(x[1].out_channels) for x in self.cnns_down))

        # self.cnns_down = torch.nn.Sequential(list(torch.nn.Sequential(x[0],x[1]) for  x in self.cnns_down))
        self.first_up = torch.nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.cnns_up = torch.nn.ModuleList(list(torch.nn.ModuleList([None, None, None]) for x in range(3)))
        self.cnns_up[0][0] = torch.nn.Conv2d(base * 4, base * 2, kernel_size=3)
        self.cnns_up[0][1] = torch.nn.Conv2d(base * 2, base * 2, kernel_size=3)
        self.cnns_up[0][2] = torch.nn.ConvTranspose2d(base * 2, base, 3, 3)
        self.cnns_up[1][0] = torch.nn.Conv2d(base * 2, base * 2, kernel_size=3)
        self.cnns_up[1][1] = torch.nn.Conv2d(base * 2, base * 2, kernel_size=3)
        self.cnns_up[1][2] = torch.nn.ConvTranspose2d(base * 2, base * 2, 3, 3)
        self.cnns_up[2][0] = torch.nn.Conv2d(base * 2, base * 2, kernel_size=3)
        self.cnns_up[2][1] = torch.nn.Conv2d(base * 2, base * 2, kernel_size=3)
        self.batch_norms_up = torch.nn.ModuleList(list(torch.nn.BatchNorm2d(x[1].out_channels) for x in self.cnns_up))
        self.final_up = torch.nn.ConvTranspose2d(base * 2, num_classes, 3, 3)
        # self.cnns_up = torch.nn.Sequential(list(torch.nn.Sequential(x[0],x[1],x[2]) for  x in self.cnns_down))

    def forward(self, x):
        original_size = x.shape[2:]
        torch.nn.functional.interpolate(x, size=(224, 224))
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        skips = []
        for i in range(len(self.cnns_down)):
            if i > 0:
                inp = torch.cat((x, torch.zeros_like(x)), 1)
            layer_tuple = self.cnns_down[i]
            x = layer_tuple[0](x)
            x = layer_tuple[1](x)
            x = self.batch_norms_down[i](x)
            if i > 0:
                x = x + torch.nn.functional.interpolate(inp, size=x.shape[2:])
            if i != len(self.cnns_down) - 1:
                skips.append(x)
            x = torch.nn.MaxPool3d(2)(x)
        x = self.first_up(x)
        for i in range(len(self.cnns_up)):
            if self.cnns_up[i][2] is not None:
                x = torch.cat((torch.nn.functional.interpolate(skips[-i - 1], x.shape[2:]), x), 1)
            inp = x
            x = self.cnns_up[i][0](x)
            x = self.cnns_up[i][1](x)
            x = self.batch_norms_up[i](x)
            inp = inp[:, :x.shape[1], :, :]
            x = x + torch.nn.functional.interpolate(inp, size=x.shape[2:])
            if self.cnns_up[i][2] is not None:
                x = self.cnns_up[i][2](x)
        x = self.final_up(x)
        x = torch.nn.Softmax(1)(x)
        return torch.nn.functional.interpolate(x, size=original_size)

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
