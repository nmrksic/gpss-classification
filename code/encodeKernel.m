function [covMatrix] = encodeKernel(KernelMatrix, dataDim)
% This function takes a matrix encoding a kernel and constructs its
% GPML covariance function; dataDim refers to input dimensionality.

% The kernel is encoded in a 'sum of products' form, with base kernels
% being SE covariance functions in different dimension.

% The i-th row of the matrix expresses the i-th product term. Its terms are
% represented as KernelMatrix(i, :), where KernelMatrix(i, j) = k means
% that the j-th member of the product term is the base Kernel SE_{k}.

 n = size(KernelMatrix, 1);
 m = size(KernelMatrix, 2);
  
 lengths = zeros(n, 1);
   
 for i = 1:n
     
     lengths(i) = nnz(KernelMatrix(i, :));
     
     if(lengths(i)==0) % n should contain the actual number of product terms
         n = i-1;
         break;
     end
     
 end
 
 nonZeros = sum(lengths);
 
 baseCovs = cell(n, m);
 covFunctions = cell(n, 1);
 
 % we first create all the necessary base SE kernels in the baseCovs matrix
 
 masks = eye(dataDim); % to be used with covMasks
 
 for i = 1:n
     for j = 1:lengths(i)
        baseCovs{i, j} = {@covMask, { masks( KernelMatrix(i, j), : ) , {@covSEiso}   }  };
     end
 end
 
 if (nonZeros == 1) % base kernel - no need for covProd or covSum
     covMatrix = baseCovs{1, 1};
     return;
 end
 
 for i = 1:n
     % if any of these are not a product, they should not be encoded as such
     if(lengths(i) > 1)
        covFunctions{i} =  { @covProd,  baseCovs(i, 1:lengths(i))  }; %brackets?
     else
         covFunctions{i} = baseCovs{i, 1};
     end
     
 end
 
 if( size(lengths, 1) < 2 || lengths(2)==0) % single product term - no need for covSum
     covMatrix = covFunctions{i};
     return;
 end
 
 covMatrix = {@covSum,  { covFunctions{1:n} }  };

 