import torch.nn as nn
import torch.nn.functional as F

# MTL for all attributes

class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder, dropout_rate):
        super().__init__()
        self.image_encoder = image_encoder
        
        # Shared layers for all attributes
        self.shared_layers = nn.Sequential(
            nn.Linear(image_encoder_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Attribute-specific layers
        self.attribute_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 128),  # Adjust sizes as if needed
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, size)  # Final layer for each attribute
            ) for size in attribute_sizes
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)
                
    def forward(self, x):
        x = self.image_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the image features
        x = self.shared_layers(x)
        outputs = [predictor(x) for predictor in self.attribute_predictors]
        return outputs
