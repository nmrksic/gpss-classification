function [bestScore, bestHyp] = evaluateKernel(X, Y, encoderMatrix, numRestarts, inferenceMethod, likelihoodFunction, searchCriterion)
% This method finds the best kernel hyperparameters (by the designated score, 
% and returns that score and the hyperparameters used to achieve it.
% This is a local function: for cluster use parallel_bases.m and nextKernel.m
%
%   Nikola Mrksic
%   March 2014
%

    sampleSize = size(X, 1);

    [bestScore, bestHyp, ~, ~] = random_restarts(X, Y, encoderMatrix, numRestarts, inferenceMethod, likelihoodFunction, searchCriterion);  

    % if we are using BIC to guide the search, compute it from nlml:
    if searchCriterion == 0 
    
        bestScore = BIC(bestScore, encoderMatrix, sampleSize);
   
    end % otherwise, we've already obtained negative testing acc. as bestScore


end

