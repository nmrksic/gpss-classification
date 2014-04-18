function [] = runAll ()
% must be run from the gpss-classification/code folder. 
% all produced real-world data set results should be in results/store/ folder
% the synthetic experiments will be saved in
% results/SyntheticDataEvaluation/ as summary0.mat and summary1.mat ...

% first run synthetic experiments - four of them, both with likErf and likMix

numRestarts = 5;

seed = 0;

runParallel = 1; 

warning off
    addpath(genpath('./'));
warning on

try
    evaluateSyntheticData ( numRestarts, seed, runParallel, 0 ); %likErf
    system('touch results/synth1done');
catch
end



try
    evaluateSyntheticData ( numRestarts, seed, runParallel, 1 ); %likMix
    system('touch results/synth2done');
catch 
end


likelihoodFunction={@likMix,{@likUni,@likErf}};

system('mkdir -p results');
system('mkdir -p results/store');
system('mkdir -p results/progress');
system('mkdir -p results/store/likmix');

% ------------------------------------------------------------------------

system('mkdir -p results/store/likmix/r_heart');

load('../Data/classification/r_heart.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, likelihoodFunction, 10, 1, 'r_heart', 0); 
    system('touch results/progress/likmixheart');

catch 
end

system ('mv results/r_heart results/store/likmix/r_heart'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/likmix/r_pima');

load('../Data/classification/r_pima.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, likelihoodFunction, 10, 1, 'r_pima', 0); 
    system('touch results/progress/likmixpima');
    
catch 
end

system ('mv results/r_pima results/store/likmix/r_pima'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/likmix/r_breast');

load('../Data/classification/r_breast.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, likelihoodFunction, 10, 1, 'r_breast', 0); 
    system('touch results/progress/likmixbreast');
    
catch 
end

system ('mv results/r_breast results/store/likmix/r_breast'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/likmix/r_liver');

load('../Data/classification/r_liver.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, likelihoodFunction, 10, 1, 'r_liver', 0); 
    system('touch results/progress/likmixliver');

catch 
end

system ('mv results/r_liver results/store/likmix/r_liver'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/crossvalidate');

% ########################################################################

system('mkdir -p results/store/crossvalidate/r_heart');

load('../Data/classification/r_heart.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, @likErf, 10, 1, 'r_heart', 1); 
    system('touch results/progress/cvheart');

catch 
end

system ('mv results/r_heart results/store/crossvalidate/r_heart'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/crossvalidate/r_pima');

load('../Data/classification/r_pima.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, @likErf, 10, 1, 'r_pima', 1); 
    system('touch results/progress/cvpima');
    
catch 
end

system ('mv results/r_pima results/store/crossvalidate/r_pima'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/crossvalidate/r_breast');

load('../Data/classification/r_breast.mat');
try

    evaluateGPC(X, y, 5, @infLaplace, @likErf, 10, 1, 'r_breast', 1); 
    system('touch results/progress/cvbreast');

catch 
end

system ('mv results/r_breast results/store/crossvalidate/r_breast'); 

% ------------------------------------------------------------------------

system('mkdir -p results/store/crossvalidate/r_liver');

load('../Data/classification/r_liver.mat');

try
    
    evaluateGPC(X, y, 5, @infLaplace, @likErf, 10, 1, 'r_liver', 1); 
    system('touch results/progress/cvliver');

catch 
end

system ('mv results/r_liver results/store/crossvalidate/r_liver'); 

% ------------------------------------------------------------------------


% move data, then run with different likelihood mode




