import torch
import torch.nn as nn
import torchvision.models as models

class ResNet1D(nn.Module):
    """
    ResNet1D adapts a pre-trained ResNet50 model to accept 1D input and perform
    classification tasks, with optional dropout and batch normalization.

    Parameters:
    - input_channels (int): Number of input channels (features).
    - num_classes (int): Number of output classes for classification.
    - dropout_rate (float): Dropout rate to apply before the fully connected layer.
    - use_batch_norm (bool): Whether to use batch normalization after the convolutional layers.
    """
    def __init__(self, input_channels, num_classes, dropout_rate=0.5, use_batch_norm=True):
        super(ResNet1D, self).__init__()
        self.num_classes = num_classes
        # Load a pre-trained ResNet50 model with explicit weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Modify the first convolutional layer to handle 1D input
        self.resnet.conv1 = nn.Conv2d(
            in_channels=input_channels, 
            out_channels=64, 
            kernel_size=(7, 1), 
            stride=(2, 1), 
            padding=(3, 0), 
            bias=False
        )

        # Optionally add batch normalization after each residual block
        if use_batch_norm:
            self.add_batch_norm()

        # Dropout layer before the final fully connected layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Replace the fully connected layer with a new layer for the desired number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def add_batch_norm(self):
        """
        Add batch normalization layers to the model.
        """
        for name, module in self.resnet.named_children():
            if isinstance(module, nn.Sequential):
                # Iterate over the layers in the sequential block
                for layer_name, layer in module.named_children():
                    if isinstance(layer, nn.Conv2d):
                        # Create a new sequential block with the Conv2d and BatchNorm2d
                        num_channels = layer.out_channels
                        bn_layer = nn.BatchNorm2d(num_channels)

                        # Replace the original Conv2d layer with a sequential block (Conv2d + BatchNorm2d)
                        setattr(module, layer_name, nn.Sequential(layer, bn_layer))

    def forward(self, x):
        """
        Forward pass for the ResNet1D model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_channels, sequence_length).

        Returns:
        - torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Add a singleton dimension to make the input shape compatible with ResNet's 2D convolutions
        # Input shape should be (batch_size, input_channels, sequence_length, 1)
        x = x.unsqueeze(-1)

        # Forward through the modified ResNet
        x = self.resnet(x)

        # Apply dropout before the final fully connected layer
        x = self.dropout(x)

        return x

def DANWithPretrainedResNet1D(input_channels, num_classes, dropout_rate=0.5, use_batch_norm=True):
    """
    Create an instance of ResNet1D adapted for domain adaptation using a pre-trained ResNet50.

    Parameters:
    - input_channels (int): Number of input channels (features).
    - num_classes (int): Number of output classes for classification.
    - dropout_rate (float): Dropout rate to apply before the fully connected layer.
    - use_batch_norm (bool): Whether to use batch normalization.

    Returns:
    - model (ResNet1D): An instance of the ResNet1D model.
    """
    return ResNet1D(input_channels=input_channels, num_classes=num_classes, 
                    dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)
