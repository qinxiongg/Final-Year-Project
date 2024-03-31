import torch.nn as nn


# implemented SE blocks into conv2

class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        
        for layer_name in ['layer3', 'layer4']:
            layer = getattr(image_encoder, layer_name)
            for block in layer:
                block.conv2 = nn.Sequential(
                    block.conv2,
                    SEBlock(block.conv2.out_channels)
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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)