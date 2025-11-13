%%Creates and returns the encoder and decoder layers for the autoencoder. 
function [layers] = createLayers() 
    % Encoder Layers
encoderLayers = [
    imageInputLayer([32 32 3], "Name", "input")
    
    % Encoder block 1: 32x32x3 -> 16x16x32
    convolution2dLayer(3, 32, "Padding", 1, "Name", "encoder_conv1")
    reluLayer("Name", "encoder_relu1")
    convolution2dLayer(3, 32, "Padding", 1, "Name", "encoder_conv2")
    reluLayer("Name", "encoder_relu2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "encoder_pool1")
    
    % Encoder block 2: 16x16x32 -> 8x8x64
    convolution2dLayer(3, 64, "Padding", 1, "Name", "encoder_conv3")
    reluLayer("Name", "encoder_relu3")
    convolution2dLayer(3, 64, "Padding", 1, "Name", "encoder_conv4")
    reluLayer("Name", "encoder_relu4")
    maxPooling2dLayer(2, "Stride", 2, "Name", "encoder_pool2")
    
    % Encoder block 3: 8x8x64 -> 4x4x128 (bottleneck)
    convolution2dLayer(3, 128, "Padding", 1, "Name", "encoder_conv5")
    reluLayer("Name", "encoder_relu5")
    convolution2dLayer(3, 128, "Padding", 1, "Name", "encoder_conv6")
    reluLayer("Name", "encoder_relu6")
    maxPooling2dLayer(2, "Stride", 2, "Name", "encoder_pool3")
];

% Decoder Layers
decoderLayers = [
    % Decoder block 1: 4x4x128 -> 8x8x64
    transposedConv2dLayer(2, 128, "Stride", 2, "Name", "decoder_deconv1")
    reluLayer("Name", "decoder_relu1")
    convolution2dLayer(3, 64, "Padding", 1, "Name", "decoder_conv1")
    reluLayer("Name", "decoder_relu2")
    
    % Decoder block 2: 8x8x64 -> 16x16x32
    transposedConv2dLayer(2, 64, "Stride", 2, "Name", "decoder_deconv2")
    reluLayer("Name", "decoder_relu3")
    convolution2dLayer(3, 32, "Padding", 1, "Name", "decoder_conv2")
    reluLayer("Name", "decoder_relu4")
    
    % Decoder block 3: 16x16x32 -> 32x32x3
    transposedConv2dLayer(2, 32, "Stride", 2, "Name", "decoder_deconv3")
    reluLayer("Name", "decoder_relu5")
    convolution2dLayer(3, 16, "Padding", 1, "Name", "decoder_conv3")
    reluLayer("Name", "decoder_relu6")
    convolution2dLayer(3, 3, "Padding", 1, "Name", "decoder_output")
    sigmoidLayer("Name", "decoder")
 ];

% Combine encoder and decoder
layers = [encoderLayers; decoderLayers];
