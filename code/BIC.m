function bic_value = BIC(nlml, covFunctionEncoderMatrix, datasize)

bic_value = 2 * nlml + effectiveParams(covFunctionEncoderMatrix) * log (datasize);

aic_value =  2 * nlml + 2 * effectiveParams(covFunctionEncoderMatrix); % AIC penalises less (each eff. param is + 2, not +2 * log(datasize)

bic_value = aic_value;

% # parameters of sum = sum of # params of each term 
% for a product term, all variances can be conflated to a single parameter,
% so a product term of n elements has just n + 1 hyperparameters

% flexible function . py - effective operators example
