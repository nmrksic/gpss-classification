function [bestBIC, bestHyp, BicVals] = evaluateKernel(encoderMatrix, X, Y, numExp, inferenceMethod)
% this function applies three different approximations (VB, EP, Laplace)
% and penalises the score using the BIC to return the value and the approximation 
% used.


covFunction = encodeKernel(encoderMatrix, size(X, 2)); 

 sampleSize = size(X, 1);

 [BestNLML, bestHyp, ~, nlmlvals] = random_restarts(covFunction, X, Y, inferenceMethod, numExp); % just use EP. 
 
 BicVals = zeros( size(nlmlvals) );
 
 bestBIC = BIC(BestNLML, encoderMatrix, sampleSize);
 
 msg = ['The best training set BIC value achieved was: ', num2str(bestBIC)];
 
 %disp(msg)
   
end

