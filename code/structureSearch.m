function [bestBicVals, trainAccuracies, testAccuracies, hyperParameters, encoderMatrices, finalName, finalBIC, finalAcc, finalHyperParam, finalEncoder] = structureSearch(X, y, Xtst, ytst, searchDepth, numRestarts, runParallel, inferenceMethod, likelihoodFunction, searchCriterion, backtrack, optimisation)
 % Gaussian Process Structure Search for GP classification, exploring a 'sum-of-products' SE base grammar. 
 % Returns statistics on the kernels on the search path - final kernel
 % stats are also returned separately. 
 
 %   X, y, Xtst, ytst:    training and testing data, respectively. 
 %   searchDepth:         the maximum depth of the search tree.
 %   numRestarts:         the number of random restarts to be performed for finding optimal hyperparameters for each kernel evaluated. 
 %   runParallel:         0 - run search locally; 1 - run search on the cluster.
 %   inferenceMethod:     Laplace approximation used for marginal likelihood inference by default - EP is slow, VB exhibits sub-optimal perfomance. 
 %   likelihoodFunction:  error function used as the default likelihood likMix is supported as well.
 %   searchCriterion:     0 - use BIC to guide search. 1 - use cross-validated test accuracy.
 %
 %   Features not tested exhaustively (if activated):
 %
 %   backtrack:           0 - do not backtrack. 1 - backtrack during the search procedure. 
 %   optimisation:        0 - no feature selection. 1 - limit search to top 'reducedCount' base functions (by the search criterion, either BIC or test accuracy)
 %
 %
 %   Nikola Mrksic
 %   April 2013
 %
 

%%  ------  Initialise all the 'meta' variables: could be moved to a config file.  -----------

    reducedCount = 10; % the number of dimensions to use for feature selection
    meanfunc = @meanConst;
    dataDim = size(X, 2); % Dimensionality of the data (number of base kernels)
    dataSize = size(X, 1);
    
    if(nargin < 12)
        disp('Not doing feature selection.')
        optimisation = 0; % do not do feature selection by default
    end
    
    if (nargin < 11)
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
        numRestarts = 2;
    end

    if (nargin < 5)
        searchDepth = 2 * size (X, 2); % 2 * dimensionality of the data. 
    end
    
    
    encoderMatrices = cell(searchDepth, 1); % and their (sum of product) matrix representation.
    bestBicVals = zeros(searchDepth, 1); % ... and the BIC value they achieved.
    hyperParameters = cell(searchDepth, 1);    % ... together with the hyperparameters used to achieve it. 
    trainAccuracies = zeros(searchDepth, 1); % as well as the training ...
    testAccuracies = zeros(searchDepth, 1);  % ... and testing accuracies. 
    visited = zeros(searchDepth, 1);     % in case we want to include backtracking in the search (i.e. for BAM)
    kernelScores = zeros(searchDepth, 1); % the score for a kernel: originally BIC, can be cross-validated accuracy as an alternative. 
    
    
 
 %% ------------------- Create and evaluate the base kernels -------------------------------------------------------------------------------------

 
    if (runParallel == 1) % evaluate on the cluster

        system('rm -f scripts/*'); % ensure scripts left post-mortem interrupted executions do not affect execution
        [kernelScores, hyperParameters, encoderMatrices] = parallel_bases(X, y, numRestarts, inferenceMethod, likelihoodFunction, searchDepth, searchCriterion); % modify to allow different search modes. 
        
        % initialise the base kernels, determine their BIC values (on cluster) and calculate training and test accuracies:
        for i = 1:dataDim 

            covFunction = encodeKernel(encoderMatrices{i}, dataDim);

            [~,~,~,~,lp1] = gp(hyperParameters{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X, ones(size(y)));
            [~,~,~,~,lp2] = gp(hyperParameters{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, Xtst, ones(size(ytst)));

            trainAccuracies(i) = calculateAcc(lp1, X, y);
            testAccuracies(i) = calculateAcc(lp2, Xtst, ytst);
            bestBicVals(i) =   gp(hyperParameters{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);
            bestBicVals(i) = BIC(bestBicVals(i), encoderMatrices{i}, dataSize);

        end

    else

        disp( ['Evaluating the base SE kernels, number of experiments to run is ', num2str(numRestarts * dataDim),'. The number of dimensions is ', num2str(dataDim), '.'] );
        
        % initialise the base kernels and determine their BICs and training/test accuracies locally:
        for i = 1:dataDim 

            encoderMatrix = zeros(10, 10); 
            encoderMatrix(1,1) = i;
            encoderMatrices{i} = encoderMatrix;

            covFunction = encodeKernel(encoderMatrices{i}, dataDim);

            [kernelScores(i), hyperParameters{i}] = evaluateKernel( X, y, encoderMatrices{i}, numRestarts, inferenceMethod, likelihoodFunction, searchCriterion);
            [~,~,~,~,lp1] = gp(hyperParameters{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X, ones(size(y)));
            [~,~,~,~,lp2] = gp(hyperParameters{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, Xtst, ones(size(ytst)));

            trainAccuracies(i) = calculateAcc(lp1, X, y); 
            testAccuracies(i) = calculateAcc(lp2, Xtst, ytst);
            bestBicVals(i) =   gp(hyperParameters{i}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);
            bestBicVals(i) = BIC(bestBicVals(i), encoderMatrices{i}, dataSize);

        end

    end
    
    % The number of kernels so far discovered during the search process:
    currentIter = dataDim; 
    
    % Initialise the values for the base kernel to expand (the best one)
    [~, idx] = min(kernelScores(1:dataDim) );
    bestScore = kernelScores(idx);
    finalAcc = testAccuracies(idx);
    finalHyperParam = hyperParameters{idx};
    finalEncoder = encoderMatrices{idx};  
    finalBIC = bestBicVals(idx);
    finalName = decodeKernelName(finalEncoder);
    
  
    msg1 = ['BIC values of base kernels: ', num2str( bestBicVals(1:dataDim)' ) ];          
    msg2 = [' Train accuracies:           ', num2str(trainAccuracies(1:dataDim)')];      
    msg3 = [' Test accurracies:           ',  num2str(testAccuracies(1:dataDim)') ];   
    disp(msg1); disp(msg2); disp(msg3);
    fprintf(' ---------------------------------------------------------------------------------------------------------------- \n'); % and a delimiter for sake of legibility.
 
    
    
  %% -----------  Feature Selection - elimination of all but the 10 top dimensions (w.r.t. their BIC values) -----------------

  
    selectFeatures = (1:dataDim)'; % by default, no feature selection, use all dimensions

    if (optimisation == 1 && dataDim > reducedCount) % pick top base kernels (by BIC)

        selectFeatures = zeros(reducedCount, 5); 

        [sortedScores, indexes] = sort(kernelScores(1:(dataDim))); % Sort the kernels by BIC values.

        encoderMatricesNew = cell(searchDepth, 1);
        bestKernelScoreNew = zeros(searchDepth, 1);  

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

 while currentIter < searchDepth
    
    currentIter = currentIter + 1;
    
    [~, indexes] = sort(kernelScores(1:(currentIter-1))); % Determine the best kernel not yet expanded. 
    
    idx = 1; % choose the index of the kernel to expand:    
    
    for i = 1: (currentIter - 1)
        if ( visited(indexes(i) ) == 0)
            idx = indexes(i);
            visited( indexes(i) ) = 1;
            break;
        end
    end
    
    % Print the structure and hyperparameters of the kernel we are about to expand:
    disp(['Kernel being expanded at this stage of the search is: ',  decodeKernelName(finalEncoder) ,'.']);
    startingHyperparameters = hyperParameters{idx} %#ok
    
    
    % Perform a step of the search to choose the next kernel:
    [encoderMatrices{currentIter}, kernelScores(currentIter), hyperParameters{currentIter}] = nextKernel(X, y, encoderMatrices{idx} , hyperParameters{idx}, numRestarts, runParallel,  inferenceMethod, likelihoodFunction, searchCriterion, selectFeatures);
    
    
    % Compute accuracies and BIC: 
    covFunction = encodeKernel( encoderMatrices{currentIter} , dataDim); % obtain the actual kernel
    
    [~,~,~,~,lp1] = gp(hyperParameters{currentIter}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, X, ones(size(y)));
    [~,~,~,~,lp2] = gp(hyperParameters{currentIter}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y, Xtst, ones(size(ytst)));
  
    trainAccuracies(currentIter) = calculateAcc(lp1, X, y);
    testAccuracies(currentIter) = calculateAcc(lp2, Xtst, ytst);
    nlml =  gp(hyperParameters{currentIter}, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);
    bestBicVals(currentIter) = BIC(nlml, encoderMatrices{currentIter}, dataSize);
    
    
    % Print the new kernel's stats:
    msg1 = ['Best kernel constructed at this stage of the search is: ', decodeKernelName(  encoderMatrices{currentIter} ), '.'];
    msg2 = ['BIC value: ', num2str(bestBicVals(currentIter)),  '; Training accuracy: ', num2str(trainAccuracies(currentIter)), '; Test accurracy: ',  num2str(testAccuracies(currentIter)),'. Previous best BIC: ', num2str(bestScore), ' and new kernel score is: ', num2str(kernelScores(currentIter))];
    disp(msg1);     disp(msg2);   fprintf('\n ---------------------------------------------------------------------------------------------------------------- \n'); % improve log legibility.
    
    
    % Check whether we've improved on the best score we had so far; if not - stop the procedure.
    if kernelScores(currentIter) - bestScore > - 0.001 % needed for guiding cross-validation. 
            fprintf('\n Search unable to construct a better kernel. About to check backtracking condition. \n');

        if backtrack == 0
           
            % Undo the value assignments performed: 
            trainAccuracies(currentIter) = 0;
            testAccuracies(currentIter) = 0;
            bestBicVals(currentIter) =  0;
            kernelScores(currentIter) = 0;  %#ok
            hyperParameters{currentIter} = [];
            encoderMatrices{currentIter} = [];
            
            % and then stop the search:
            return;
            
        else 
            fprintf('\n Search unable to construct a better kernel. \n');
        end

    end
    
    % Assign the values of the new, currently best kernel we've found:
    bestScore = kernelScores(currentIter);
    finalAcc = testAccuracies(currentIter);
    finalHyperParam = hyperParameters{currentIter};
    finalEncoder = encoderMatrices{currentIter}; 
    finalBIC = bestBicVals(currentIter);
    finalName = decodeKernelName(finalEncoder);
    
 end
 
 
end
