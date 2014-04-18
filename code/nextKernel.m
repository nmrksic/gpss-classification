function [bestKernelEncoder, bestScoreVal, bestKernelHyper] = nextKernel( X, y, kernelMatrix, kernelHypers, numRestarts, runParallel, inferenceMethod, likelihoodFunction, searchCriterion, selectFeatures)
%
% This function expands the provided matrix representation of a covariance
% function by trying to add another base function or by multiplying any product 
% term with a base function. The number of expansion operators performed is: 
% dim * (1 + number_of_prodterms), at each stage of the search.
%
%   Nikola Mrksic
%   April 2014
%

    fileID = fopen('randomRestartScript.m', 'r');
    scriptCode = fscanf(fileID, '%s');
    fclose(fileID);

    meanfunc = @meanConst;
    dataSize = size(X, 1);
    dataDim = size(X, 2);

    if (nargin < 8)
        
        selectFeatures = zeros(dataDim, 1); % expand all dimensions if no feature selection is done.
        for i = 1:dataDim
            selectFeatures (i) = i;
        end
    end

    if (nargin < 7)
        likelihoodFunction = @likErf;
    end

    kernelCount = 0; % number of new kernels that we will try at this stage

    kernelNames = cell(1, 1000); % remembers the names of the kernel we want to construct
    restartPositions = zeros(1, 1000); % remember, for each new kernel, what 2 values we will be restarting.
    % represents the index of the element after which we are to insert the 2
    % new hyperparameters (which are to be varied)

    if (nargin < 6)
        inferenceMethod = @infLaplace;
    end

    if (nargin < 5)
        numRestarts = 2;
    end

    n = size(kernelMatrix, 1);

    lengths = zeros(n, 1);

    for i = 1:n
        lengths(i) = nnz(kernelMatrix(i, :));

        if(lengths(i)==0) % n should contain the actual number of product terms
            n = i-1;
            break;
        end
    end

    encoderMatrices = cell(dataDim * ( 1 + n ) ); % this is where we store the expanded kernels matrices
    dimensionExpanded = zeros( (n + 1) * dataDim, 1); % used to remember which dimension that specific kernel is adding, in order to initialize with proper lengthscales. 

    bestScoreVal = inf;  % set the best value of the BIC achieved to infinity.  
    bestKernelEncoder = full(kernelMatrix); % ... and remember the form of the best kernel.
    bestKernelHyper = [];

 % First, try all dataDim kernels which correspond to old kernel + SE_{i}
 
     for i = 1:dataDim

         if ( ismember( i, selectFeatures) ) % if this dimension is considered

             nextCovFun = full(kernelMatrix); % make a fresh copy of the new matrix.
             nextCovFun(n+1, 1) = i;    % ... and add a new basis function

             kernelCount = kernelCount  +  1;
             restartPositions(kernelCount) = nnz(kernelMatrix) * 2; % after all the terms we already have

             encoderMatrices{i} = nextCovFun;
             dimensionExpanded(i) = i;

         end

     end
 
     % Then, try multiplying each product term with (any) basis function:
     
     for basis = 1:dataDim  % for all basis functions...
         
         if ( ismember( basis, selectFeatures) ) % if this dimension is considered
             
             for i = 1:n        % ... and for all terms in the summation:
                 
                 nextCovFun = full(kernelMatrix); % fresh copy of the new matrix.
                 nextCovFun(i, lengths(i)+1) = basis;    % ... multiply i-th term with SE_{basis}
                 
                 kernelCount = kernelCount  +  1;
                 restartPositions(kernelCount) = 2 * nnz( kernelMatrix(1:i, :) ); % all rows including this one
                 
                 encoderMatrices{ kernelCount } = nextCovFun;
                 dimensionExpanded( kernelCount ) = basis;

             end
             
         end
     end
 
     % To the restart function we provide the kernel structure, and we construct
     % kernelCount new hyperparameter arrays and feed them all to the restart
     % function, after restarting along the position of the new basis kernel.
     
     % the numbers of hyperparameters that the kernels have
     newHyperCount = nnz(kernelMatrix) * 2 + 2;
     hyperParameters = cell(1, kernelCount);
     
     for i = 1:kernelCount
         
         newHyper = zeros( newHyperCount, 1 );
         temp = kernelHypers.cov;
         newHyper (1 : newHyperCount - 2  ) = temp(1 : newHyperCount - 2 );
         hyperParameters{i} = newHyper;
         
     end

 
 % Now, having created the kernelCount covfunctions and their initial
 % hyperparameters, we need to create kernelCount * numExp matlab scripts
 % that will restart and optimise these hyperparameters:
 
 % We first initialize the new hyperparameters at their positions:
 
    [minDist, maxDist] = lengthscales(X); % Find the minimum, maximum characteristic length scales for each dimension in the data set. 

    initialisedHyperParams = zeros ( numRestarts * kernelCount, newHyperCount); % 1..numExp first Kernel, (numExp+1)...(2*numExp) second Kernel, and so on...


    for i = 1:kernelCount
        
        hyperParsOld = hyperParameters{i};
        
        for j = 1:numRestarts
            
            idx = (i - 1) * numRestarts + j;
            
            initialisedHyperParams ( idx , : ) = hyperParsOld;
            
            initialisedHyperParams( idx , (restartPositions(i) + 1)  : (restartPositions(i)+2) ) =  log( [minDist( dimensionExpanded(i) ) + (maxDist( dimensionExpanded(i) ) - minDist( dimensionExpanded(i) )  ) * rand(), 1 + 5 * rand()] );
            
            if ( restartPositions(i) + 2 < newHyperCount) % we need to shift everything after restarted positions by 2
                
                initialisedHyperParams ( idx, restartPositions(i) + 3 : newHyperCount ) = hyperParsOld ( restartPositions (i) + 1 : newHyperCount - 2 );
                
            end
            
        end
    end
         
    % Now that we have the hyperparameters, all that is left is to call
    % minimize and then evaluate NLML for all of these. To do so, we need to
    % send separate scripts to the cluster.
    
    % We now create the final hyperparams that are to be sent to the cluster:
    
    finalHyperParams = cell(1,  numRestarts * kernelCount );
        
    for i = 1 : (numRestarts * kernelCount)
        finalHyperParams{i}.mean = 0;
        finalHyperParams{i}.cov = initialisedHyperParams (i, :);
        
        % if it's a likelihood mixture (currently the only non - likErf
        % likelihood supported, initialise its hyperparameters:
        if  iscell(likelihoodFunction) == 1
            finalHyperParams{i}.lik = [ -1 + randn(), 1 + randn() ];
        end
        
    end
        
    bicValues = zeros(numRestarts * kernelCount, 1);
    crossValidatedAccuracies = zeros(numRestarts * kernelCount, 1);
    hyperParameters = cell(1,  numRestarts * kernelCount); % data to be received from the cluster
 
 
 % Perform random restarts and collect output data:
 
  if ( runParallel == 1)
     system('mkdir -p scripts'); % make sure we have the scripts folder available. 

     for i = 1:(kernelCount)
         for j = 1:numRestarts
             
             idx = (i-1)*numRestarts + j;
             
            if ( iscell(likelihoodFunction)==0 ) % if not a cell, then it's likErf, so no hyperparameters. 
                 
                script_code = ['i=', num2str(idx), ';likelihoodFunction=@likErf;', 'covFunction=', decodeKernelCode(encoderMatrices{i}, size(X,2)), ';hyperParameters.mean=0;hyperParameters.cov=[', num2str(finalHyperParams{idx}.cov), '];',  scriptCode];
                
             else
                 
                script_code = ['i=', num2str(idx), ';likelihoodFunction={@likMix,{@likUni,@likErf}};', 'covFunction=', decodeKernelCode(encoderMatrices{i}, size(X,2)), ';hyperParameters.mean=0;hyperParameters.cov=[', num2str(finalHyperParams{idx}.cov), '];hyperParameters.lik=[', num2str(finalHyperParams{idx}.lik) ,'];', scriptCode];
             
             end
             
             fileID = fopen(['scripts/script', sprintf('%09d', idx), '.m'], 'w');
             fprintf(fileID, '%s\n', script_code);
             fclose(fileID);
             
         end
     end
     
     save('data/data.mat', 'X', 'y');
     
     [~, ~] = system('python runscriptsinparallel.py'); % Supress output from the cluster.  
     
     % Load output data:
     
     for i = 1:(kernelCount)
         for j = 1 : numRestarts
             
             idx = (i-1)*numRestarts + j;
             covFunction = encodeKernel( encoderMatrices{i},  dataDim );

             file_name = ['outputs/script' sprintf('%09d', idx) '.mat'];
             load(file_name);
             
             hyperParameters{ idx } = hypN; %#ok - this is what we just loaded.
             bicValues( idx ) = BIC(bicValue, encoderMatrices{i}, dataSize ); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)
             
             crossValidatedAccuracies(idx) = crossValidatedAccuracy(X, y, covFunction, hyperParameters{idx}, inferenceMethod, likelihoodFunction);
             
         end
     end
     
     system('rm outputs/*');

      elseif ( runParallel == 0)

          for i = 1:(kernelCount)
              for j = 1 : numRestarts

                  idx = (i-1)*numRestarts + j;
                  covFunction = encodeKernel( encoderMatrices{i},  dataDim );

                  if(size(X, 1) > 250)
                      subset = randsample(dataSize, 250);
                      hyperParams = minimize(finalHyperParams{idx}, @gp, -300, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X(subset, :), y(subset));
                      hypN = minimize(hyperParams, @gp, -30, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);
                  else
                      hypN = minimize(finalHyperParams{idx}, @gp, -300, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);
                  end

                  nlml = gp(hypN, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);

                  bicValues(idx) = BIC(nlml, encoderMatrices{idx}, dataSize ); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)
                  hyperParameters{idx} = hypN;

                  crossValidatedAccuracies(idx) = crossValidatedAccuracy(X, y, covFunction, hyperParameters{idx}, inferenceMethod, likelihoodFunction);

              end
          end
  end
       
  
    % Depending on the search criterion, assign kernelScores and return the
    % best one back to the structureSearch call:
  
      
    bestScoreList = zeros( size(bicValues ) );
  
    if searchCriterion == 0
        bestScoreList = bicValues;
    elseif searchCriterion == 1
        bestScoreList = crossValidatedAccuracies;
    end
    
    
    [bestScoreVal, bestId] = min(bestScoreList);

    trueIdx = int32(floor((bestId - 0.0001) / numRestarts) + 1);

    bestKernelEncoder = encoderMatrices{trueIdx};

    bestKernelHyper = hyperParameters{bestId};
 
 
end
 
