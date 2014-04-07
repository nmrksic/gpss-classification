
% Assumes you're in the Matlab folder

system('mkdir -p outputs');
system('mkdir -p data');
system('mkdir -p scripts');
system('mkdir -p images');

addpath ( genpath ( './' )); 

load( '../Data/classification/r_liver.mat' )

evaluateGPC( X, y, 3, @infLaplace, 10, 1);

