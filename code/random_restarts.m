function [minimalNLML, minHyp, hyp, nlml] = random_restarts(covFunction, X, Y, inferenceMethod, numExp)
% non-fear random restarts for a single covariance function - no longer compatible with Automated Statistician

hyp = cell(numExp);
nlml = zeros(1, numExp);

 likfunc = @likErf;
 meanfunc = @meanConst; 
 
 numHyper = eval(feval(covFunction{:})); % number of (effective) hyperparams

 [minDist, maxDist] = lengthscales(X); %Find the minimum, maximum characteristic length scale in the data
    
for i = 1:numExp
 
    Hyp.mean = 0;
    Hyp.cov = rand(1, numHyper) * 10+1;% increase potential starting values for the sf - length scales changed anyway 
    Hyp.cov(1:2:(numHyper-1)) = minDist + (maxDist - minDist) * rand(1, numHyper / 2); % length-scales
    
    Hyp.cov = log (Hyp.cov);
    
    hypN = minimize(Hyp, @gp, -1000, inferenceMethod, meanfunc, covFunction, likfunc, X, Y); % 300 a good value. 50 * numHyper!
    hyp{i} = hypN;
    nlml(i) = gp(hypN, inferenceMethod, meanfunc, covFunction, likfunc, X, Y);

end

[minimalNLML, minIndex] = min(nlml); % it doesn't hurt to leave nlml until later as it doesn't change what the minimum is: 
% all of them are multiplied by two and added the same factor (numHyper * nData)

minHyp = hyp{minIndex};

% 10 seemed to be enough for the paper
