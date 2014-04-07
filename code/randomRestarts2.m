function [bicValue, hypN] = randomRestarts2(covFunction, hyperParameters, X, Y, inferenceMethod)
% returns the BIC value and the new hyperparameters post minimisation
% TO BE REVISED in accordance with random restart script...

 likfunc = @likErf;
 meanfunc = @meanConst; 
    
 hypN = minimize(hyperParameters, @gp, -1000, inferenceMethod, meanfunc, covFunction, likfunc, X, Y); % 300 a good value. 50 * numHyper!
 nlml = gp(hypN, inferenceMethod, meanfunc, covFunction, likfunc, X, Y);

 sampleSize = size(X, 1); 
 
 bicValue = BIC(nlml, covFunction, sampleSize);
 
