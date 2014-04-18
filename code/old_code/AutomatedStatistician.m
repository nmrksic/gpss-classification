function [bestBicVals, trainAccuracies, testAccuracies, bestHyper, bestName, bestScore, bestAcc, bestHyperParam, finalEncoder, encoderMatrices] = StructureSearch(X, y, X_tst, y_tst, searchSteps, numExp, runParallel, inferenceMethod, likelihoodFunction, searchCriterion, backtrack, optimisation)
 % GPSS for GP classification, with a covSE grammar and search defined as in
 % the original paper (operations: +B, *B, substitute any term with B). 

%%  ------  Initialise all the 'meta' variables: could be moved to a config file.  -----------

    reducedCount = 10; % the number of dimensions to use for feature selection
    meanfunc = @meanConst;
    dataDim = size(X, 2); % Dimensionality of the data (number of base kernels)

    if(nargin < 11)
        disp('Not doing feature selection.')
        optimisation = 0; % do not do feature selection by default
    end
    
    if (nargin < 10)
        backtrack = 0; % do not backtrack by default
    end
 
    if (nargin < 10)
        searchCriterion = 0; % 0: BIC; 1: cross-validated accuracy. 
    end
    
    if (nargin < 9)
        likelihoodFunction = @likErf;
    end
 
    if (nargin < 8)
        inferenceMethod = @infLaplace;
    end

    if (nargin < 7)
        runParallel = 0; % do not run in parallel automatically.
    end
 
    if (nargin < 6)
        numExp = 2;
    end

    if (nargin < 5)
        searchSteps = 2 * size (X, 2); % 2 * dimensionality of the data. 
    end
    
    
    encoderMatrices = cell(1, searchSteps); % and their (sum of product) matrix representation.
    bestBicVals = zeros(1, searchSteps); % ... and the BIC value they achieved.
    bestHyper = cell(1, searchSteps);    % ... together with the hyperparameters used to achieve it. 
    trainAccuracies = zeros(1, searchSteps); % as well as the training ...
    testAccuracies = zeros(1, searchSteps);  % ... and testing accuracies. 
    visited = zeros(1, searchSteps);     % in case we want to include backtracking in the search (i.e. for BAM)
    kernelScores = zeros(1, searchSteps); % the score for a kernel: originally BIC, can be cross-validated accuracy as an alternative. 
    
    
 
 %% ------------------- Create and evaluate the base kernels -------------------------------------------------------------------------------------

 
    if (runParallel == 1) % evaluate on the cluster

        system('rm -f scripts/*'); % ensure scripts left post-mortem interrupted executions do not affect execution
        [kernelScores, bestHyper, encoderMatrices] = parallel_bases(X, y, numExp, inferenceMethod, likelihoodFunction, searchSteps, searchCriterion);
        
        % initialise the base kernels, determine their BIC values (on cluster) and calculate training and test accuracies:
        for i = 1:dataDim 

            covFunction = encodeKernel(encoderMatrices{i}, dataDim);

            [~,~,~,~,lp1] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X, ones(size(y)));
            [~,~,~,~,lp2] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X_tst, ones(size(y_tst)));

            trainAccuracies(i) = calculateAcc(lp1, X, y);
            testAccuracies(i) = calculateAcc(lp2, X_tst, y_tst);
            bestBicVals(i) =   gp(bestHyper{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);

        end

    else

        disp( ['Evaluating the base SE kernels, number of experiments to run is ', num2str(numExp * dataDim),'. The number of dimensions is ', num2str(dataDim)] );
        
        % initialise the base kernels and determine their BICs and training/test accuracies locally:
        for i = 1:dataDim 

            encoderMatrix = zeros(10, 10); 
            encoderMatrix(1,1) = i;
            encoderMatrices{i} = encoderMatrix;

            covFunction = encodeKernel(encoderMatrices{i}, dataDim);

            [kernelScores(i), bestHyper{i}] = evaluateKernel( encoderMatrices{i}, X, y, numExp, inferenceMethod, searchCriterion); %modify function to take this !!!!!!!!!!!!!!!

            [~,~,~,~,lp2] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X_tst, ones(size(y_tst)));
            [~,~,~,~,lp1] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X, ones(size(y)));
            
            trainAccuracies(i) = calculateAcc(lp1, X, y); 
            testAccuracies(i) = calculateAcc(lp2, X_tst, y_tst);
            bestBicVals(i) =   gp(bestHyper{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);

        end

    end
    
    % The number of kernels so far discovered during the search process:
    currentIter = dataDim; 
    
    % Initialise the values for the base kernel to expand (the best one)
    [~, idx] = min(kernelScores(1:dataDim) );
    bestScore = kernelScores(idx);
    bestAcc = testAccuracies(idx);
    bestHyperParam = bestHyper{idx};
    finalEncoder = squeeze(encoderMatrices{idx});  
    bestName = decodeKernelName(finalEncoder);
    
    
    msg1 = ['BIC values of base kernels: ', num2str(bestBicVals(1:dataDim)) ];          
    msg2 = [' Train accuracies:           ', num2str(trainAccuracies(1:dataDim))];      
    msg3 = [' Test accurracies:           ',  num2str(testAccuracies(1:dataDim)) ];   
    disp(msg1); disp(msg2); disp(msg3);
    fprintf(' ---------------------------------------------------------------------------------------------------------------- \n'); % and a delimiter for sake of legibility.
 
    if searchCriterion == 0
        kernelScores = bestBicVals;
    elseif searchCriterion == 1
        kernelScores = -testAccuracies; % as we pick the minimal value
    end
    
  
    
  %% -----------  Feature Selection - elimination of all but the 10 top dimensions (w.r.t. their BIC values) -----------------

  
    selectFeatures = (1:dataDim)'; % by default, no feature selection, use all dimensions

    if (optimisation == 1 && dataDim > reducedCount) % pick top base kernels (by BIC)

        selectFeatures = zeros(reducedCount, 5); 

        [sortedScores, indexes] = sort(kernelScores(1:(dataDim))); % Sort the kernels by BIC values.

        encoderMatricesNew = cell(1, searchSteps);
        bestKernelScoreNew = zeros(1, searchSteps);  

        % auxiliary variables
        for i = 1:reducedCount

            encoderMatricesNew{i} = encoderMatrices{indexes(i)};
            bestKernelScoreNew(i) = sortedScores(i); 
            selectFeatures(i) = indexes(i);

        end

        encoderMatrices = encoderMatricesNew;
        kernelScores = bestKernelScoreNew;

        currentIter = reducedCount;
        disp(['Feature Selection completed: search space reduced to top ', num2str(reducedCount), ' dimensions in terms of BIC.']); 

    end

    if (optimisation ==1 && dataDim <= reducedCount)
        disp(' Not doing feature selection: there are <= ', num2str(reducedCount), ' dimensions in the data set.');
    end

    
    
%%  ------------------------------------------ Start the structure search --------------------------------------------------------------------------

 while currentIter < searchSteps
    
    currentIter = currentIter + 1;
    
    [currMin, indexes] = sort(kernelScores(1:(currentIter-1))); % Determine the best kernel not yet expanded. 
    
    idx = 1; % choose the index of the kernel to expand:    
    
    for i = 1: (currentIter - 1)
        if ( visited(indexes(i) ) == 0)
            idx = indexes(i);
            visited( indexes(i) ) = 1;
            break;
        end
    end
    
    % Print the structure and hyperparameters of the kernel we are about to expand:
    disp(['Kernel being expanded at this stage of the search is: ',  decodeKernelName(finalEncoder{idx}) ,'.']);
    startingHyperparameters = bestHyper{idx} %#ok
    
    
    % Perform a step of the search to choose the next kernel:
    [encoderMatrices{currentIter}, kernelScores(currentIter), bestHyper{currentIter}] = nextKernel( encoderMatrices{idx} , bestHyper{idx}, dataDim, X, y, numExp, runParallel,  inferenceMethod, searchCriterion, selectFeatures);
    
    
    % Compute accuracies and BIC: 
    covFunction = encodeKernel( encoderMatrices{currentIter} , dataDim); % obtain the actual kernel
    
    [~,~,~,~,lp1] = gp(bestHyper{currentIter}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X, ones(size(y)));
    [~,~,~,~,lp2] = gp(bestHyper{currentIter}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X_tst, ones(size(y_tst)));
  
    trainAccuracies(currentIter) = calculateAcc(lp1, X, y);
    testAccuracies(currentIter) = calculateAcc(lp2, X_tst, y_tst);
    bestBicVals(currentIter) =  gp(bestHyper{currentIter}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);

    
    % Print the new kernel's stats:
    msg1 = ['Best kernel constructed at this stage of the search is: ', decodeKernelName(  encoderMatrices{currentIter} ), '.'];
    msg2 = ['BIC value: ', num2str(bestBicVals(currentIter)),  '; Training accuracy: ', num2str(trainAccuracies(currentIter)), '; Test accurracy: ',  num2str(testAccuracies(currentIter)),'.'];
    disp(msg1);     disp(msg2);   fprintf('\n ---------------------------------------------------------------------------------------------------------------- \n'); % improve log legibility.
    
    
    % Check whether we've improved on the best score we had so far; if not - stop the procedure.
    if kernelScores(currentIter) > currMin(1) 

        if backtrack == 0
            fprintf('\n Search unable to construct a better kernel. Stopping the search procedure. \n');
            
            % Undo the value assignments performed: 
            trainAccuracies(currentIter) = 0;
            testAccuracies(currentIter) = 0;
            bestBicVals(currentIter) =  0;
            kernelScores(currentIter) = 0;  %#ok
            bestHyper{currentIter} = [];
            encoderMatrices{currentIter} = [];
            
            % and then stop the search:
            return;
            
        else 
            fprintf('\n Search unable to construct a better kernel. \n');
        end

    end
    
    % Assign the values of the new, currently best kernel we've found:
    bestScore = kernelScores(currentIter);
    bestAcc = testAccuracies(currentIter);
    bestHyperParam = bestHyper{currentIter};
    finalEncoder = encoderMatrices{currentIter}; 
    bestName = decodeKernelName(finalEncoder);
    
 end

 