%% Will import the images and preprocess them(if needed). 
function [images] = getTrainingImages() 
    %%Get image directory. 
    dir = 'cifar-10-batches-mat'; 
    data = load(fullfile(dir, 'data_batch_1.mat')).data; 
    %%Reshape images for training. 
    X = reshape(data', 32, 32, 3, []); 
    X = permute(X, [2, 1, 3, 4]); 
    images = double(X) / 255.0; 
    %%Use this if using the built-in autoencoder. 
    %%images = squeeze(num2cell(X, [1 2 3]))';
