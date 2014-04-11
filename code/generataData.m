function [trnX, trnY, tstX, tstY] = generateData(X, y)

if (nargin < 3)
    numExp = 2;
end

if nargin < 4
    inferenceMethod = @infLaplace;
end

if nargin < 5
    numFolds = 10;
end

if nargin < 6
    runParallel = 0; % do not run in parallel automatically
end

if nargin < 7
    expName = 'crossValidatedExperiment'; % this should be called with, for example, 'r_liver'
end

    seed = 0;

    InitialiseRand(seed); % random seed initialised here. 

    averageAcc = 0;
    
    testAccuracies = zeros(numFolds, 1);
    bicValues = zeros(numFolds, 1);
    kernelNames = cell(numFolds, 1);
    finalHypers = cell (numFolds, 1); 
    trainAccs = cell(numFolds, 1);
    finalEncoders = cell(numFolds, 1);

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, numFolds); % numFolds-fold cross validation data sets
    
end