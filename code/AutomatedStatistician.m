function [kernelNames, bestBicVals, predictiveAccuraccies, bestHyper, trainAccuracies, bestName, bestBic, bestAcc, bestHyperParam, finalEncoder] = AutomatedStatistician(X, y, X_tst, y_tst, searchSteps, numExp, runParallel, inferenceMethod, backtrack, optimisation)
 % GPSS for GP classification, with a covSE grammar and search defined as in
 % the original paper (operations: +B, *B, substitute any term with B). 

 % Kernels are represented in the 'sum of products' form, with each kernel represented by a matrix.  

    bestName = 'fail';
    bestBic = inf;
    bestAcc = inf;
    bestHyperParam = [100 100];
    finalEncoder = [1 2; 3 4];

 if(nargin < 10)
     disp('Not doing feature selection.')
     optimisation = 0; % do not do feature selection by default
 end
    
 if (nargin < 9)
     backtrack = 0; % do not backtrack by default
 end
 
 
 if (nargin < 8)
     inferenceMethod = @infLaplace;
 end

 if (nargin < 7)
     runParallel = 0; % do not run in parallel automatically
 end
 
 if (nargin < 6)
     numExp = 2;
 end


 if (nargin < 5)
     searchSteps = 2 * size (X, 2); % 2 * dimensionality of the data. 
 end

 
 reducedCount = 10; % the number of dimensions to use for feature selection
 meanfunc = @meanConst;
 likfunc = @likErf;
 
 dim = size(X, 2); % Dimensionality of the data (number of base kernels)

 encoderMatrices = zeros(searchSteps, 10, 10); 
 covFunctions = cell(1, searchSteps); % The list of base and subsequently chosen kernels
 kernelNames = cell(1, searchSteps);  % Names of the kernels chosen. 
 bestBicVals = zeros(1, searchSteps); % ... and the BIC value they achieved
 visited = zeros(1, searchSteps);

 bestHyper = cell(1, searchSteps);
 predictiveAccuraccies = zeros(1, searchSteps);
 trainAccuracies = zeros(1, searchSteps); 
 
 % ------------------- Create base kernels ---------------------------------------------
 
 % To parallelise, we need only to call parallel_bases instead of the code
 % between these lines. 
 
  
 if (runParallel == 1)
     system('rm -f scripts/*.m');
     parallel_bases
 else
     
     disp( ['Evaluating the base SE kernels, number of experiments to run is ', num2str(numExp * dim),'.'] );

     for i = 1:dim % initialise the base kernels and determine their BICs and test accuracies

         encoderMatrices(i, 1, 1) = i;
         covFunctions{i} = encodeKernel(encoderMatrices(i, :, :), dim);
         kernelNames{i} = ['SE', num2str(i)];

         [bestBicVals(i), bestHyper{i}] = evaluateKernel( squeeze( encoderMatrices(i, :, :)) , X, y, numExp, inferenceMethod);
         
         [~,~,~,~,lp] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunctions{i}, likfunc, X, y, X_tst, ones(size(y_tst)));
         predictiveAccuraccies(i) = calculateAcc(lp, X_tst, y_tst);

         % train accs, alternative search criterion:
         [~,~,~,~,lp2] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunctions{i}, likfunc, X, y, X, ones(size(y)));
         trainAccuracies(i) = calculateAcc(lp2, X, y); 
         % ....

     end
     
 end
 
  % --------------------------------------------------------------------------------

 
  msg = ['BIC values of base kernels: ', num2str(bestBicVals(1:dim)) ];
 disp(msg);
 
 msg = [' Train accuracies:           ', num2str(trainAccuracies(1:dim))];
 disp(msg);
 
 msg = [' Test accurracies:           ',  num2str(predictiveAccuraccies(1:dim)) ];
 disp(msg);
 disp(' ');
 disp('---------------------------------------------------------------------------------------------------------------------');
 disp(' ');
 

 
 currentIter = dim; % the index of the kernel we are currently choosing.
 
 selectFeatures = (1:dim)'; % by default, no feature selection, use all dimensions
 
 if (optimisation == 1 && dim > reducedCount) % pick top base kernels (by BIC)
    
    selectFeatures = zeros(reducedCount, 5); 
     
     [sortedBIC, indexes] = sort(bestBicVals(1:(dim))); % Sort the kernels by BIC values.
    
    encoderMatricesNew = zeros(searchSteps, 10, 10);
    covFunctionsNew = cell(1, searchSteps); 
    bestBicValsNew = zeros(1, searchSteps);  

     % auxiliary variablesnextKernelParallel(squeeze( encoderMatrices(idx, :, :) ), bestHyper{idx}, dim, X, y, numExp, inferenceMethod, selectFeatures);
    for i = 1:reducedCount
        encoderMatricesNew(i, :, :) = encoderMatrices( indexes(i), :, :);
        covFunctionsNew{i} = covFunctions { indexes(i) }; 
        kernelNamesNew{i} = kernelNames {indexes(i)}; 
        bestBicValsNew(i) = sortedBIC(i); 
        selectFeatures(i) = indexes(i);
        
    end
    
    
    encoderMatrices = encoderMatricesNew;
    covFunctions = covFunctionsNew;
    kernelNames = kernelNamesNew;
    bestBicVals = bestBicValsNew;
 
    currentIter = reducedCount;
    disp(['Feature Selection completed - search space reduced to 5 dimensions:', kernelNames{1:reducedCount}] );

 end
 
 if (optimisation ==1 && dim <= reducedCount)
     disp('Not doing feature selection - there are <= ', reducedCount, ' dimensions of the data.');
 end

 while currentIter < searchSteps
    
    currentIter = currentIter + 1; 
         
    [currMin, indexes] = sort(bestBicVals(1:(currentIter-1))); % Determine the best kernel not yet expanded. % CHANGE2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    idx = 1; % index of the kernel to expand:
    
    for i = 1: (currentIter - 1)
        if ( visited(indexes(i) ) == 0)
            idx = indexes(i);
            visited( indexes(i) ) = 1;
            break;
        end
    end
    
    if (bestBic == inf)  % corrected bug when single base is the best solution
        bestName = kernelNames{idx};
        bestBic = bestBicVals(idx);
        bestAcc = predictiveAccuraccies();
        
        bestHyperParam = bestHyper{idx};
        finalEncoder = squeeze(encoderMatrices(idx, :, :));    

    end
    
    disp(['Kernel being expanded at this stage of the search is: ', kernelNames{idx},'.']);
    startingHyperparameters = bestHyper{idx}
    
    [encoderMatrices(currentIter, :, :), bestBicVals(currentIter), bestHyper{currentIter}] = nextKernel(squeeze( encoderMatrices(idx, :, :) ), bestHyper{idx}, dim, X, y, numExp, runParallel,  inferenceMethod, selectFeatures);

    kernelNames{currentIter} = decodeKernelName(squeeze(encoderMatrices(currentIter, :, :)));

    covFunctions{currentIter} = encodeKernel( squeeze( encoderMatrices(currentIter, :, :) ), dim); % obtain the actual kernel
    


    % train accuracies: 
    [~,~,~,~,lp2] = gp(bestHyper{currentIter}, inferenceMethod, meanfunc, covFunctions{currentIter}, likfunc, X, y, X, ones(size(y)));
    trainAccuracies(currentIter) = calculateAcc(lp2, X, y);
    
    % test accuracies:
    [~,~,~,~,lp] = gp(bestHyper{currentIter}, inferenceMethod, meanfunc, covFunctions{currentIter}, likfunc, X, y, X_tst, ones(size(y_tst)));
    predictiveAccuraccies(currentIter) = calculateAcc(lp, X_tst, y_tst);

    msg = ['Best kernel constructed at this stage of the search is: ', kernelNames{currentIter},'.'];
    disp(msg);
    msg = ['BIC value: ', num2str(bestBicVals(currentIter)),  '; Training accuracy: ', num2str(trainAccuracies(currentIter)), '; Test accurracy: ',  num2str(predictiveAccuraccies(currentIter)),'.'];
    disp(msg);
    disp(' ');
    disp('---------------------------------------------------------------------------------------------------------------------');
    disp(' ');
    
    
    if bestBicVals(currentIter) > currMin(1) % in this case, we are not creating a better kernel (at least in terms of the BIC)
            disp('Search unable to construct a better kernel. ')
    
        if backtrack == 0
             disp('Stopping the search procedure. ')
             return;
        end
    end
    
    bestName = kernelNames{currentIter};
    bestBic = bestBicVals(currentIter);
    bestAcc = predictiveAccuraccies(currentIter);
    bestHyperParam = bestHyper{currentIter};
    finalEncoder = squeeze(encoderMatrices(currentIter, :, :));    
    
 end

 