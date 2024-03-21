import torch.nn as nn
import torch

# CBAM implementation

class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(image_encoder, layer_name)
            for block in layer:
                block.conv2 = nn.Sequential(
                    block.conv2,    
                    CBAM(block.conv2.out_channels)
                )
        
        self.attribute_sizes = attribute_sizes
        self.attribute_predictors = nn.ModuleList(
            [nn.Linear(image_encoder_output_dim, size) for size in attribute_sizes])
        # Apply Kaiming initialization to the attribute predictors
        for predictor in self.attribute_predictors:
            nn.init.kaiming_normal_(predictor.weight, nonlinearity='relu')
            nn.init.zeros_(predictor.bias)

    def predict_from_features(self, x):
        # Predict each attribute
        outputs = [predictor(x) for predictor in self.attribute_predictors]
        return outputs

    def forward(self, x):
        x = self.image_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the image features
        outputs = self.predict_from_features(x)
        return outputs

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out) 
    
