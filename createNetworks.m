%%Trains the two networks(one with post processing and the other with post
%%processing and writes them to two MAT files for use in SIMULINK. 
function [network] = createNetworks() 
    params = getModelParams(); 
    training_images = getTrainingImages(); 
    layers = createLayers(); 

    network = trainnet(training_images, training_images, dlnetwork(layers), "mse", params); 
    %%network = trainAutoencoder(training_images, 1024, 'MaxEpochs',200, 'UseGPU', true); 