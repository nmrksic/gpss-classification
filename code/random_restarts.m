function [minimalScore, minHyp, hyperList, nlmlList, accuracyList] = random_restarts(X, y, encoderMatrix, numRestarts, inferenceMethod, likelihoodFunction, searchCriterion)
% this function finds the best kernel hyperparameters (by the designated score, and returns
% the score and the parameters used to achieve that score.
%
%
%   Nikola Mrksic
%   April 2014
%

    hyperList = cell(numRestarts);
    nlmlList = zeros(1, numRestarts);
    accuracyList = zeros(1, numRestarts); 

    meanfunc = @meanConst; 
    covFunction = encodeKernel(encoderMatrix, size(X, 2)); 

    numHyper = nnz(encoderMatrix);
    
    [minDist, maxDist] = lengthscales(X); %Find the minimum, maximum characteristic length scale in the data
    
    for i = 1:numRestarts

        hyperParameters.mean = 0;
        hyperParameters.cov = rand(1, 2 * numHyper) * 5 + 1;% initialize starting values for the sf: length scales changed anyway 
        
        encoderTranspose = encoderMatrix';
        dimensionEncoded = encoderTranspose((encoderTranspose>0));
        
        % Initialize hyperparameters: 
        for d = 1:numHyper
        
            hyperParameters.cov( 2 * d - 1 ) = minDist( dimensionEncoded(d) ) + ( maxDist( dimensionEncoded(d) ) - minDist( dimensionEncoded(d) ) ) * rand(); 
        
        end
            
        hyperParameters.cov = log (hyperParameters.cov);
        
        % if it's a likelihood mixture (currently the only non-likErf
        % likelihood supported, initialise its hyperparameters:
        if  ( iscell(likelihoodFunction) == 1 )
            hyperParameters.lik = [ -1 + randn(), 1 + randn() ];
        end

        if(size(X, 1) > 250)
            subset = randsample(size(X, 1), 250);
            hyperParameters = minimize(hyperParameters, @gp, -300, inferenceMethod, @meanConst, covFunction, likelihoodFunction, X(subset, :), y(subset));
            hypN = minimize(hyperParameters, @gp, -30, inferenceMethod, @meanConst, covFunction, likelihoodFunction, X, y);
        
        else
            
            hypN = minimize(hyperParameters, @gp, -300, inferenceMethod, @meanConst, covFunction, likelihoodFunction, X, y);
        
        end

        hyperList{i} = hypN;
        nlmlList(i) = gp(hypN, inferenceMethod, meanfunc, covFunction, likelihoodFunction, X, y);
        accuracyList(i) = crossValidatedAccuracy(X, y, covFunction, hypN, inferenceMethod, likelihoodFunction);
        
        % --------- Prune small lengthscales -----------------------
             newHyperParamAll = hypN.cov;
             newHyperParam = newHyperParamAll(1:2:end); % take only lengthscales
             
             encoderDims = encoderMatrix;
             encoderDims = encoderDims';
             
             encoderDims = encoderDims(encoderDims > 0);
             
             for i2 = 1:size(encoderDims) % if any of the dimensions went under, stop considering this kernel
                
                 if exp( newHyperParam (i2) ) < minDist ( encoderDims ( i2 ) ) 
                    nlmlList(i) = 2000000000;
                    accuracyList(i) = 0;                 
                 end
             end
        
        % ----------------------------------------------------------
        
    end

    % if we are using BIC, return the best NLML value:
    if searchCriterion == 0
        
        [minimalScore, minIndex] = min(nlmlList); % it doesn't hurt to use nlml here: it doesn't change what the minimum is (we are evaluating the same kernel numRestarts times) 
        minHyp = hyperList{minIndex};
        
    elseif searchCriterion == 1
        
        [minimalScore, minIndex] = min(accuracyList); % it doesn't hurt to use nlml here: it doesn't change what the minimum is (we are evaluating the same kernel numRestarts times) 
        minHyp = hyperList{minIndex};
        
    end


