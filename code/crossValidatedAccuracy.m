function [Accuracy] = crossValidatedAccuracy(X, y, covFunction, hyperParameters, inferenceMethod, likelihoodFunction)
% X and y are the training set, which we split into 10 folds and then
% compute the average accuracy of the GP supplied to the function:
%
%
%   Nikola Mrksic
%   April 2014
%

    meanfunc = @meanConst;
    InitialiseRand(0);

    [Xtrn, ytrn, Xtst, ytst] = crossValidate(X, y, 10);

    Accuracy = 0;
    
    for i = 1:10
        
        [~,~,~,~,lp] = gp(hyperParameters, inferenceMethod, meanfunc, covFunction, likelihoodFunction, Xtrn{i}, ytrn{i}, Xtst{i}, ones(size(ytst{i})));
        Accuracy = Accuracy + calculateAcc(lp, Xtst{i}, ytst{i});
        
    end
    
    Accuracy = - Accuracy / 10; % return the mean cross-validated accuracy (negative to enable search by minimal value

