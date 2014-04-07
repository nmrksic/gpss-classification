function [numParams] = effectiveParams(covFuncEncoderMatrix)

% As we are dealing with a sum of products form, the number of hyperparams
% is number of non-zero elements (lengthscales) + number of nonzero rows
% (signal variances)

lengthscales = nnz(covFuncEncoderMatrix);

signalvars = 0;

for i = 1: size(covFuncEncoderMatrix, 1)
    if (covFuncEncoderMatrix(i, 1) == 0)
        break;
    end
    signalvars = i;
end

numParams = lengthscales + signalvars; % for pure covSEiso kernels, this is the figure

