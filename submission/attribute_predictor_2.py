import torch.nn as nn
import torch.nn.functional as F

class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder, nucleus_shape_index):
        super().__init__()
        self.image_encoder = image_encoder
        self.attribute_sizes = attribute_sizes
        self.nucleus_shape_index = nucleus_shape_index  # Index of the nucleus shape attribute
        self.attribute_predictors = nn.ModuleList()
        
        for i, size in enumerate(attribute_sizes):
            if i == nucleus_shape_index:
                # Add task-specific layers for nucleus shape
                self.attribute_predictors.append(nn.Sequential(
                    nn.Linear(image_encoder_output_dim, 256),  # Task-specific layer
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, size)  # Final layer for nucleus shape
                ))
            else:
                # Regular predictor for other attributes
                self.attribute_predictors.append(nn.Linear(image_encoder_output_dim, size))
        
        # Apply Kaiming initialization to the attribute predictors
        for predictor in self.attribute_predictors:
            if isinstance(predictor, nn.Linear):
                nn.init.kaiming_normal_(predictor.weight, nonlinearity='relu')
                nn.init.zeros_(predictor.bias)
            else:  # For Sequential modules
                nn.init.kaiming_normal_(predictor[-1].weight, nonlinearity='relu')
                nn.init.zeros_(predictor[-1].bias)

    def predict_from_features(self, x):
        outputs = [predictor(x) for predictor in self.attribute_predictors]
        return outputs

    def forward(self, x):
        x = self.image_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the image features
        outputs = self.predict_from_features(x)
        return outputs
