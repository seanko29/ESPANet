# HFDKDModel - Heterogeneous Feature Distillation for Knowledge Distillation

## Overview

The `HFDKDModel` is a knowledge distillation model designed for super-resolution tasks that can handle heterogeneous architectures (e.g., ViT teacher and CNN student). It implements a two-stage training process:

1. **Feature Alignment Stage**: Aligns teacher features to student feature space
2. **Knowledge Distillation Stage**: Trains the student with aligned features

## Common Error: `'HFDKDModel' object has no attribute 'net_t'`

This error occurs when the teacher network (`net_t`) is not properly initialized. This typically happens due to missing configuration.

### Root Cause

The error occurs because:
1. The `network_t` configuration is missing from your config file
2. The `path.pretrain_network_t` is not specified
3. The configuration structure is incorrect

### How to Fix

1. **Add Teacher Network Configuration**:
   ```yaml
   network_t:
     type: SwinIR  # or your teacher network type
     # ... other teacher network parameters
   ```

2. **Add Teacher Pretrained Weights Path**:
   ```yaml
   path:
     pretrain_network_t: /path/to/teacher/weights.pth
   ```

3. **Ensure Complete Configuration Structure**:
   ```yaml
   model_type: HFDKDModel
   network_t:
     type: SwinIR
     # ... parameters
   network_g:
     type: EDSR
     # ... parameters
   path:
     pretrain_network_t: /path/to/teacher/weights.pth
     pretrain_network_g: /path/to/student/weights.pth  # optional
   ```

### Example Configuration

See `examples/hfdkd_example_config.yaml` for a complete example configuration.

### Validation

The model now includes automatic configuration validation. If you see validation errors, check that all required fields are present in your configuration.

## Key Features

- **Heterogeneous Architecture Support**: Works with different teacher and student architectures
- **Multi-scale Feature Alignment**: Aligns features at multiple scales
- **Frequency-aware Processing**: Uses FFT for better high-frequency recovery
- **Dynamic Weighting**: Adjusts loss weights during training
- **Two-stage Training**: Alternating alignment and distillation stages

## Usage

1. Prepare your configuration file following the example
2. Ensure teacher and student networks are properly configured
3. Set the correct paths to pretrained weights
4. Run training as usual

## Configuration Parameters

### Required
- `network_t`: Teacher network configuration
- `network_g`: Student network configuration  
- `path.pretrain_network_t`: Path to teacher weights

### Optional (with defaults)
- `train.hfd_opt.teacher_dim`: 512
- `train.hfd_opt.student_dim`: 64
- `train.hfd_opt.num_scales`: 3
- `train.hfd_opt.lambda_pixel`: 1.0
- `train.hfd_opt.lambda_feat`: 10.0
- `train.hfd_opt.lambda_custom`: 10.0
- `train.hfd_opt.lambda_freq`: 0.1
- `train.hfd_opt.alignment_frequency`: 5
- `train.hfd_opt.use_dynamic_weights`: true

## Troubleshooting

1. **Configuration Validation Error**: Check that all required fields are present
2. **Teacher Network Not Found**: Ensure `network_t` and `path.pretrain_network_t` are configured
3. **Feature Extraction Error**: Some networks may not have `get_features` method - the model will use fallback methods
4. **Memory Issues**: Reduce batch size or use smaller networks

## Architecture Components

- **HeterogeneousFeatureAligner**: Aligns ViT features to CNN space
- **FrequencyProjector**: Projects features in frequency domain
- **TokenToFeatureMap**: Converts ViT tokens to spatial features
- **SRReconstructionHead**: Reconstruction head for alignment stage
- **HFDSRLoss**: Comprehensive loss function for HFD-SR 