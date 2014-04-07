function [averageAcc, testAccuracies, bicValues, kernelNames] = evaluateGPC(X, y, numReruns, inferenceMethod, numFolds, runParallel)

if (nargin < 3)
    numReruns = 2;
end

if nargin < 4
    inferenceMethod = @infLaplace;
end

if nargin < 5
    numFolds = 2;
end

if nargin < 6
    runParallel = 0; % do not run in parallel automatically
end

    InitialiseRand(4); % random seed initialised here. 

    averageAcc = 0;
    
    testAccuracies = zeros(numFolds, 1);
    bicValues = zeros(numFolds, 1);
    kernelNames = cell(numFolds, 1);
    trainAccs = cell(numFolds, 1);
    [trnX, trnY, tstX, tstY] = crossValidate(X, y, numFolds); % numFolds-fold cross validation data sets
    
    dim = size(trnX{1}, 1); %dimensionality of the data - used to determine number of runs. For now, set to a low multiple (say 2, max 3)
    
    number_of_runs = 2 * dim;
    
    for i = 1:numFolds
   
        disp(['Currently evaluating ', num2str(i), '/', num2str(numFolds), ' cross validation data sets.']);
        
         [~, ~, ~, ~, trainAccs{i}, kernelNames{i}, bicValues(i), testAccuracies(i) ] = ...
                 AutomatedStatistician(trnX{i}, trnY{i} , tstX{i}, tstY{i}, number_of_runs , numReruns, runParallel, inferenceMethod, 0, 0);
         
         averageAcc = averageAcc + testAccuracies(i);
         
         currentAverage = averageAcc / i
         
    end

    averageAcc = averageAcc / numFolds;
    
     save('data/evaluateGPCLiverEP1000.mat', 'averageAcc', 'testAccuracies', 'bicValues', 'kernelNames');

    
end