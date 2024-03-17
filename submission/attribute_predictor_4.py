# import torch.nn as nn


# class AttributePredictor(nn.Module):
#     def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder):
#         super().__init__()
#         self.image_encoder = image_encoder
#         self.attribute_sizes = attribute_sizes
#         self.attribute_predictors = nn.ModuleList(
#             [nn.Linear(image_encoder_output_dim, size) for size in attribute_sizes])
#         # Apply Kaiming initialization to the attribute predictors
#         for predictor in self.attribute_predictors:
#             nn.init.kaiming_normal_(predictor.weight, nonlinearity='relu')
#             nn.init.zeros_(predictor.bias)

#     def predict_from_features(self, x):
#         # Predict each attribute
#         outputs = [predictor(x) for predictor in self.attribute_predictors]
#         return outputs

#     def forward(self, x):
#         x = self.image_encoder(x)
#         x = x.view(x.size(0), -1)  # Flatten the image features
#         outputs = self.predict_from_features(x)
#         return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, input_dim)  # Adjusted to keep input_dim consistent

    def forward(self, x):
        # Compute attention scores & apply softmax to get weights
        attention_scores = self.attention_weights(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Element-wise multiplication of input features and attention weights
        attended = x * attention_weights  # Element-wise multiplication for weighted features
        return attended  # Return attended without summing up to preserve the input_dim for predictor

class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder, nucleus_shape_idx):
        super(AttributePredictor, self).__init__()
        self.image_encoder = image_encoder
        self.attribute_sizes = attribute_sizes
        self.attribute_predictors = nn.ModuleList([nn.Linear(image_encoder_output_dim, size) for size in attribute_sizes])
        self.nucleus_shape_idx = nucleus_shape_idx
        self.nucleus_shape_attention = AttentionLayer(image_encoder_output_dim)
        
        # Initialize the attribute predictors
        for predictor in self.attribute_predictors:
            nn.init.kaiming_normal_(predictor.weight, nonlinearity='relu')
            nn.init.zeros_(predictor.bias)

    def forward(self, x):
        x = self.image_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the image features if not already done
        
        outputs = []
        for idx, predictor in enumerate(self.attribute_predictors):
            if idx == self.nucleus_shape_idx:
                # Apply attention mechanism for nucleus shape
                attended_features = self.nucleus_shape_attention(x)
                output = predictor(attended_features)
            else:
                output = predictor(x)
            outputs.append(output)
        return outputs