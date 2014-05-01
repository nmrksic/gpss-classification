function [cvAccuracy1, cvAccuracy2, testAccuracies1, testAccuracies2] = BayesianModelAveraging(expName)

    load(['../Data/classification/', expName, '.mat']);

    seed = 0;
    InitialiseRand(seed); % random seed initialised here. 

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, 10); % numFolds-fold cross validation data sets

    filepath = 'results/Archive/runs/crossvalidationruns/';

    dims = size(X, 2);
    
    testAccuracies1 = zeros(10, 1);
    testAccuracies2 = zeros(10, 1); 
    
    for folds = 1:10
        
        cummulativeProbs  = zeros(size(tstY{folds})); 
        cummulativeLabels = zeros(size(tstY{folds}));
        
        filePathNew = [filepath, expName, '/fold', num2str(folds), '/searchStats.mat'];
        
        load(filePathNew); 
    
        BicValsList(BicValsList==0) = [];

        weights = exp(-BicValsList); 
        
        norm = sum(weights); 

        weights = weights ./ norm; % this determines the importance of each classifier

        numEl = numel(BicValsList); 

        for covId = 1:numEl 

            covF = encodeKernel(squeeze(allEncoderMatrices{covId} ), dims );

            [~,~,~,~,lp] = gp(hyperList{covId}, @infLaplace, @meanConst, covF, @likErf, trnX{folds}, trnY{folds}, tstX{folds}, ones(size(tstY{folds})));

            cummulativeProbs = cummulativeProbs + weights(covId) * lp;

            predictions = pullClasses(exp(lp), 0.5); 
            
            cummulativeLabels = cummulativeLabels + weights(covId) * predictions' ;
                                   
        end
                
        
        testAccuracies1(folds) = calculateAcc(cummulativeProbs, tstX{folds}, tstY{folds});
        testAccuracies2(folds) = calculateAcc(cummulativeLabels, tstX{folds}, tstY{folds});
        
    end
          
    cvAccuracy1 = (1 - sum(testAccuracies1) / 10) * 100;
    cvAccuracy2 = (1 - sum(testAccuracies2) / 10) * 100;
    
end