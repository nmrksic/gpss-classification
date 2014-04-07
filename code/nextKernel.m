function [bestKernel, bestBicVal, bestHypers] = nextKernel (kernelMatrix, kernelHypers, dataDim, X, y, numExp, inferenceMethod, selectFeatures)
% This function expands the provided matrix representation of a covariance
% function by trying to add another base function, multiply any term with a
% base function, and finally by trying to replace any of the existing bases
% with a new base function. 

% The number of expansion operators performed is: dim + dim * num_prodterms 
% The first operation is cheap (dim). The second one is more expensive (dim * num_prodterms).

if (nargin < 7)
    selectFeatures = zeros(dataDim, 1); % expand all dimensions if no feature selection is done. 
    for i = 1:dataDim
        selectFeatures (i) = i;
    end
end

 meanfunc = @meanConst; 
 likfunc = @likErf;
 
 if (nargin < 6)
     inferenceMethod = @infLaplace;
 end

 if (nargin < 5)
     numExp = 2;
 end

 n = size(kernelMatrix, 1);
 m = size(kernelMatrix, 2);
  
 lengths = zeros(n, 1);
 
 for i = 1:n
     lengths(i) = nnz(kernelMatrix(i, :));
     
     if(lengths(i)==0) % n should contain the actual number of product terms
         n = i-1;
         break; 
     end
 end
 
 nextCovFun = full(kernelMatrix); % This is the (current) nextKernel we are evaluating. 
 
 bestBicVal = inf;  % set the best value of the BIC achieved to infinity.  
 bestKernel = full(kernelMatrix); % ... and remember the form of the best kernel.
 bestHypers = [];

 % First, try all dataDim kernels which correspond to old kernel + SE_{i}
 
 for i = 1:dataDim
     
     if ( ismember( i, selectFeatures) ) % if this dimension is considered

         nextCovFun = full(kernelMatrix); % make a fresh copy of the new matrix.
         nextCovFun(n+1, 1) = i;    % ... and add a new basis function

         covFun = encodeKernel(nextCovFun, dataDim); % create the actual covariance function

         % in this function, we just return the best covariance function in its
         % matrix-encoded form (with its BIC). Parsing its name and finding test accuracy is done later.  

         [BicVal, Hypers] = evaluateKernel(covFun, X, y, numExp, inferenceMethod);

         if BicVal < bestBicVal
            bestBicVal = BicVal;
            bestKernel = full(nextCovFun); 
            bestHypers = Hypers;
         end
         
     end
     
 end
 
 % Then, try multiplying each product term with (any) basis function:
 
 for basis = 1:dataDim  % for all basis functions...
     
       if ( ismember( basis, selectFeatures) ) % if this dimension is considered
     
         for i = 1:n        % ... and for all product terms:

             nextCovFun = full(kernelMatrix); % fresh copy of the new matrix.
             nextCovFun(i, lengths(i)+1) = basis;    % ... multiply i-th term with SE_{basis}

             covFun = encodeKernel(nextCovFun, dataDim); 

             [BicVal, Hypers] = evaluateKernel(covFun, X, y, numExp, inferenceMethod);

             if BicVal < bestBicVal
                bestBicVal = BicVal;
                bestKernel = full(nextCovFun); 
                bestHypers = Hypers;
             end

         end
         
       end
 end
 
 
 
end
 
 