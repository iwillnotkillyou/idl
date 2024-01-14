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
        self.backbone = torchvision.models.resnet18(True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.layer2 = nn.Identity()
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        from torchvision import transforms
        self.conv1 = nn.Conv2d(64,64,1)
        self.upconv1 = nn.ConvTranspose2d(64,48,2,2)
        self.upconv2 = nn.ConvTranspose2d(48,32,2,2)
        self.upconv3 = nn.ConvTranspose2d(32,23,2,2)
        self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        
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

        x = self.backbone.conv1(x)
        print(x.shape)
        x = self.backbone.bn1(x)
        print(x.shape)
        x = self.backbone.relu(x)
        print(x.shape)
        x = self.backbone.maxpool(x)
        print(x.shape)
        x = self.backbone.layer1(x)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.upconv1(x)
        print(x.shape)
        x = self.upconv2(x)
        print(x.shape)
        x = self.upconv3(x)
        print(x.shape)


    
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