function [AlphaValues, testAccs1, testAccs2] = evaluateBMA(expName)

    load(['../Data/classification/', expName, '.mat']);

    load(['results/BMA/', expName]); %, covFunctions, Scores, Hypers); 


    AlphaValues = zeros(100, 1); 
    
    testAccs1 = zeros(100, 1); 
    testAccs2 = zeros(100, 1);
    
    seed = 0;
    InitialiseRand(seed); % random seed initialised here. 

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, 10); % numFolds-fold cross validation data sets
    useDaveFolds % Comment this if you don't want to use Dave's folds

    for alphaC = 1 : 100

        alpha = alphaC / 100;
    
        for folds = 1:10
            
            cummulativeProbs  = zeros(size(tstY{folds}));
            cummulativeLabels = zeros(size(tstY{folds}));
            
            
            allCovFuns = covFunctions{folds};
            allScores = Scores{folds};
            allHypers = Hypers{folds};
            
            
                     
            weights = exp(-alpha * allScores);
            
            norm = sum(weights);
            
            weights = weights ./ norm; % this determines the importance of each classifier
            
            for covId = 1 : numel(weights)
                
                [~,~,~,~,lp] = gp(allHypers{covId}, @infLaplace, @meanConst, allCovFuns{covId}, @likErf, trnX{folds}, trnY{folds}, tstX{folds}, ones(size(tstY{folds})));
                
                cummulativeProbs = cummulativeProbs + weights(covId) * lp;
                
                predictions = pullClasses(exp(lp), 0.5);
                
                cummulativeLabels = cummulativeLabels + weights(covId) * predictions' ;
                
            end
                        
            testAccuracies1(folds) = calculateAcc(cummulativeProbs, tstX{folds}, tstY{folds});
            testAccuracies2(folds) = calculateAcc(cummulativeLabels, tstX{folds}, tstY{folds});
            
        end
        
        AlphaValues(alphaC) = alpha;
        testAccs1(alphaC) = (1 - sum(testAccuracies1) / 10) * 100;
        testAccs2(alphaC) = (1 - sum(testAccuracies2) / 10) * 100;
  
    end
    
    filePathNew = ['results/BMA/', expName, 'alpha.mat']; % where to write the outputs

    save(filePathNew, 'AlphaValues', 'testAccs1' , 'testAccs2');
    
    
end
