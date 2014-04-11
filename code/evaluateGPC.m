function [averageAcc, testAccuracies, bicValues, kernelNames] = evaluateGPC(X, y, numExp, inferenceMethod, numFolds, runParallel, expName)

if (nargin < 3)
    numExp = 2;
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


if nargin < 7
    expName = 'crossValidatedExperiment'; % this should be called with, for example, 'r_liver'
     dimensionLabels = cell(size(x, 2), 1); 
     for i = 1 : size(x, 2)
         dimensionLabels{i} = ['Dimension ', num2str(i)]; % if label names are not provided
     end
     fprintf('No dimension labels supplied! ');
end



    seed = 0;

    InitialiseRand(seed); % random seed initialised here. 
    
    % initialiseDataDimensionLabels; % get dimension labels for breast,
    % pima, liver, heart.  % no need for this here.

    averageAcc = 0;
    
    testAccuracies = zeros(numFolds, 1);
    bicValues = zeros(numFolds, 1);
    kernelNames = cell(numFolds, 1);
    finalHypers = cell (numFolds, 1); 
    trainAccs = cell(numFolds, 1);
    finalEncoders = cell(numFolds, 1);

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, numFolds); % numFolds-fold cross validation data sets
    
    dim = size(X, 2); % dimensionality of the data - used to determine number of runs. For now, set to a low multiple (say 2, max 3)
    
    number_of_runs = 2*dim;
    
    fileprefix = ['results/', expName, '/', ];     
        
    system([' mkdir -p ', fileprefix]);

    for i = 1:numFolds
           
        currentFold = i; % declared just for the sake of saving the data
        
        disp(['Currently evaluating ', num2str(i), '/', num2str(numFolds), ' cross validation data sets.']);
        
         
        kernelSearchLog = evalc(' [kernelNamesList, BicValsList, testAccuracciesList, hyperList, trainAccuraciesList, kernelNames{i}, bicValues(i), testAccuracies(i), finalHypers{i}, finalEncoders{i}, allEncoderMatrices ] = AutomatedStatistician(trnX{i}, trnY{i} , tstX{i}, tstY{i}, number_of_runs , numExp, runParallel, inferenceMethod, 0, 0);  ') ;  
     
       %[kernelNamesList, BicValsList, testAccuracciesList, hyperList, trainAccuraciesList, kernelNames{i}, bicValues(i), testAccuracies(i), finalHypers{i}, finalEncoders{i}, allEncoderMatrices ] = ...
        % AutomatedStatistician(trnX{i}, trnY{i} , tstX{i}, tstY{i}, number_of_runs , numExp, runParallel, inferenceMethod, 0, 0); 
     
         averageAcc = averageAcc + testAccuracies(i);
         
         currentAverage = averageAcc / i
         
         filePrefixNew = [fileprefix, 'fold', num2str(currentFold) , '/'];
         
         system([' mkdir -p ', filePrefixNew]);
         
         kernelNamesList = kernelNamesList';
         BicValsList = BicValsList';
         testAccuracciesList =testAccuracciesList';
         hyperList = hyperList';
         trainAccuraciesList = trainAccuraciesList'; % transpose for the sake of saving them nicely: could be moved to AutomatedStatistician.m

         train_test_distance = trainAccuraciesList - testAccuracciesList; % will contain the difference between test and train, indicating the level of overfitting. 
         effHyperCount = zeros( size (  train_test_distance ) ); 

         numIterations = sum(nnz(BicValsList)); % the number of search stages 

         for j = 1:numIterations
            effHyperCount(j) = effectiveParams(squeeze( allEncoderMatrices(j, :, :) ) );
         end     

         save ( [ filePrefixNew, 'searchStats.mat'], 'kernelNamesList', 'BicValsList', 'trainAccuraciesList', ...
            'testAccuracciesList', 'hyperList', 'numExp', 'seed', 'expName', 'currentFold', 'train_test_distance', 'effHyperCount', 'allEncoderMatrices' );

         fileID = fopen([filePrefixNew, 'kernelSearchLog.txt'], 'w');
         fprintf(fileID, '%s\n', kernelSearchLog);
         fclose(fileID);
        
        for plott = dim + 1 : numIterations
    
             filePrefixPlot = [filePrefixNew, 'Stage', num2str(plott), '/']; % creating folder structure for the subsequent (local) run of plots
             system([' mkdir -p ', filePrefixPlot]);
             % plotPosteriors(trnX{i}, trnY{i}, squeeze( allEncoderMatrices(plott, :, :) ) , hyperList{plott}, filePrefixPlot, dimensionLabels);

         end
         
         % plotPosteriors(trnX{i}, trnY{i}, finalEncoders{i}, finalHypers{i}, filePrefixNew, dimensionLabels);
        
    end

    averageAcc = averageAcc / numFolds;
    
    save(['results/', expName, '/summary.mat'], 'averageAcc', 'testAccuracies', 'bicValues', 'kernelNames');

    
end