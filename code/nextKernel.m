function [bestKernel, bestBicVal, bestHypers] = nextKernel(kernelMatrix, kernelHypers, dataDim, X, y, numExp, runParallel, inferenceMethod, selectFeatures)
% This function expands the provided matrix representation of a covariance
% function by trying to add another base function, multiply any term with a
% base function, and finally by trying to replace any of the existing bases
% with a new base function. 

% The number of expansion operators performed is: dim + dim * num_prodterms 
% The first operation is cheap (dim). The second one is more expensive (dim * num_prodterms).

 fileID = fopen('randomRestartScript.m', 'r');
 scriptCode = fscanf(fileID, '%s');
 fclose(fileID); 

    likfunc = @likErf;
    meanfunc = @meanConst; 
 
if (nargin < 7)
    selectFeatures = zeros(dataDim, 1); % expand all dimensions if no feature selection is done. 
    for i = 1:dataDim
        selectFeatures (i) = i;
    end
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
     numExp = 2;
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
 
 encoderMatrices = cell(1, 1000); % this is where we store the expanded kernels matrices
 
 bestBicVal = inf;  % set the best value of the BIC achieved to infinity.  
 bestKernel = full(kernelMatrix); % ... and remember the form of the best kernel.
 bestHypers = [];

 % First, try all dataDim kernels which correspond to old kernel + SE_{i}
 
 for i = 1:dataDim
     
     if ( ismember( i, selectFeatures) ) % if this dimension is considered

         nextCovFun = full(kernelMatrix); % make a fresh copy of the new matrix.
         nextCovFun(n+1, 1) = i;    % ... and add a new basis function
         
         kernelCount = kernelCount  +  1;
         kernelNames{kernelCount} = decodeKernelCode(nextCovFun, dataDim); 
         restartPositions(kernelCount) = nnz(kernelMatrix) * 2; % after all the terms we already have
         encoderMatrices{i} = nextCovFun;
         
     end
     
 end
 
 % Then, try multiplying each product term with (any) basis function:
 
 for basis = 1:dataDim  % for all basis functions...
     
       if ( ismember( basis, selectFeatures) ) % if this dimension is considered
     
         for i = 1:n        % ... and for all product terms:

             nextCovFun = full(kernelMatrix); % fresh copy of the new matrix.
             nextCovFun(i, lengths(i)+1) = basis;    % ... multiply i-th term with SE_{basis}

             kernelCount = kernelCount  +  1;
             kernelNames{kernelCount} = decodeKernelCode(nextCovFun, dataDim);
             restartPositions(kernelCount) = 2 * nnz( kernelMatrix(1:i, :) ); % all rows including this one
             
             encoderMatrices{ dataDim + (basis-1) * n + i } = nextCovFun;
             
         end
         
       end
 end
 
 % Now, for all kernelCount kernels, we optimise, keeping the hypers we had
 % so far the same as before - this is hard to implement. 
 
 % Perhaps use a kernel hyper matrix similar to how we encode matrices?
 
 % Maybe easier to remember the index of the hypers for the kernel and then
 % just reset those? NumHyper is the number of elements in the encoder
 % matrix, in the same order, so when we update for each of these, we can
 % tell which two are to be reset - the others are left intact. 
 
 % Now, to a restart function we give kernel, we construct kernelCount new
 % hyperparam arrays and feed them all into restart functions - along with
 % the position they are to restart.
 
 % Moreover, we should pre-process the distances in the dataset here, and
 % pass those values into the restart function as well. 
 
 newHyperCount = nnz(kernelMatrix) * 2 + 2;
 hyperParameters = cell(1, kernelCount);
  
 for i = 1:kernelCount
   
    hyperParameters{i} = kernelHypers.cov; 
        
    newHyper = zeros( newHyperCount, 1 );
    
    temp = hyperParameters{i};
    
    newHyper (1 : newHyperCount - 2  ) = temp(1 : newHyperCount - 2 );
    
    hyperParameters{i} = newHyper;
        
 end

 
 % Now, having created the kernelCount covfunctions and their initial
 % hyperparameters, we need to create kernelCount * numExp matlab scripts
 % that will restart and optimise these hyperparameters. 
 
 % We can initialise the random hyperparameters here :
 
 [minDist, maxDist] = lengthscales(X); %Find the minimum, maximum characteristic length scale in the data

 initialisedHyperParams = zeros ( numExp * kernelCount, newHyperCount); % 1..numExp first Kernel, (numExp+1)...(2*numExp) second Kernel, and so on...
 
  covFunctions = cell (1,  numExp * kernelCount ); 
 
 for i = 1:kernelCount

     hyperParsOld = hyperParameters{i}; 
     
     for j = 1:numExp
                  
         initialisedHyperParams ( (i-1) * numExp + j , : ) = hyperParsOld;
         
         % TODO: 1/10th of sample signal variance - betting on SNR = 0.1
         initialisedHyperParams( (i-1) * numExp + j , (restartPositions(i) + 1)  : (restartPositions(i)+2) ) =  log( [minDist + (maxDist - minDist) * rand(1, 1), 10 * rand(1,1)] ); 
         
         if ( restartPositions(i) + 2 < newHyperCount) % we need to shift everything after restartPos by 2
        
            initialisedHyperParams ( (i-1) * numExp + j, restartPositions(i) + 3 : newHyperCount ) = hyperParsOld ( restartPositions (i) + 1 : newHyperCount - 2 );
         
         end
         
         covFunctions{ (i-1) * numExp + j } = eval( kernelNames{i});
         
     end
 end
         
 % Now that we have the hyperparameters, all that is left is to call
 % minimize and then evaluate NLML for all of these. To do so, we need to
 % send separate scripts to the cluster. 
 
 % We now create the final hyperparams that are to be sent to the cluster:
 
 finalHyperParams = cell(1,  numExp * kernelCount ); 

 
  for i = 1 : (numExp * kernelCount) 
      finalHyperParams{i}.mean = 0;
      finalHyperParams{i}.cov = initialisedHyperParams (i, :); 
  end
 
 
 bicValues = zeros(numExp * kernelCount, 1); 
 hyperParameters = cell(1,  numExp * kernelCount); % data to be received from the cluster
 
 % disp(['Starting expansion of the ', num2str(kernelCount), ' potential successor kernels:']);
 
 %{ 
  % to print all the initial hyperparameters or not
 for i = 1:kernelCount
     for j = 1:numExp
          disp([decodeKernelName(encoderMatrixStore{i}), ' with starting hyperparameters: ', num2str(initialisedHyperParams((i-1)*numExp + j, :)) ]);
     end
 end
 
 %}
 
 
 % ---------- Parallel call for random restarts ----------------------------
 
  if ( runParallel == 1)
      parallel_call
  end
  
   
  if ( runParallel == 0) 
 
      for i = 1:(kernelCount)
          for j = 1 : numExp
              
             if mod((i-1)*numExp + j, numExp) == 0
                % disp(['Evaluating kernel ', num2str((i-1)*numExp + j), ' of ', num2str(numExp * kernelCount), '.']);
             end

            hypN = minimize(finalHyperParams{(i-1)*numExp + j}, @gp, -1000, inferenceMethod, meanfunc, covFunctions{(i-1)*numExp + j}, likfunc, X, y);
            nlml = gp(hypN, inferenceMethod, meanfunc, covFunctions{(i-1)*numExp + j}, likfunc, X, y);

            bicValues( (i-1)*numExp + j) = BIC(nlml, encoderMatrices{(i-1)*numExp + j}, size(X, 1)); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)

            hyperParameters{(i-1)*numExp + j} = hypN;

            % train accs, alternative search criterion:
            [~,~,~,~,lp2] = gp(hyperParameters{(i-1)*numExp + j}, inferenceMethod, meanfunc, covFunctions{(i-1)*numExp + j}, likfunc, X, y, X, ones(size(y)));
            trainAccs((i-1)*numExp + j) = calculateAcc(lp2, X, y); 

          end
  
      end
    
  end
      
      
 % -------------------------------------------------------------------------

 [bestBicVal, bestId] = min(bicValues); % max(trainAccs);
  
 trueIdx = int32(floor((bestId - 0.0001) / numExp) + 1);
  
 bestKernel = encoderMatrices{ trueIdx };
 
 bestHypers = hyperParameters{bestId};
 
 % These can be uncommented for debugging purposes:
 %
 % bicvalsare = bicValues
 % bestkernelis = bestKernel
 % besthyperis = bestHypers
 % bestbicvalueis = bestBicVal
 %
 % disp(['Kernel count: ', num2str(kernelCount), ' Best exp: ' , num2str(bestId), 'Kernel name ID: ', num2str(trueIdx), ' Best kernel: ', kernelNames{bestId}]); 
 %
 
end
 
