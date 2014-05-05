function [testAccuracies] = StructureDiscoveryBMA (expName)

    load(['../Data/classification/', expName, '.mat']);
    
    encoderChosen = [2 6 1; 7 0 0];
    covF = encodeKernel(encoderChosen, size(X, 2));
    
    testAccuracies = zeros(10, 1);
    hyp.mean = 1.0850;
    hyp.cov = [0.8463 0.6937 0.0745 0.8852 1.4647 -0.5860 1.1022 0.4907];

%    hyp.mean = 0;
 %   hyp.cov = zeros(10, 1);
    
  %  hyp = minimize(hyp, @gp, -300, @infLaplace, @meanConst, covF, @likErf, X, y)
        
    seed = 0;
    InitialiseRand(seed); % random seed initialised here. 

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, 10); % numFolds-fold cross validation data sets
    
    
      for folds = 1:10

          hyp2 = minimize(hyp, @gp, -300, @infLaplace, @meanConst, covF, @likErf, trnX{folds}, trnY{folds})

          [~,~,~,~,lp] = gp(hyp2, @infLaplace, @meanConst, covF, @likErf, trnX{folds}, trnY{folds}, tstX{folds}, ones(size(tstY{folds})));

          predictions = pullClasses(exp(lp), 0.5);
                        
          testAccuracies(folds) = calculateAcc(predictions, tstX{folds}, tstY{folds})
            
      end
      
      testAccuracies = 1 - testAccuracies;
      averageAccuracy = sum(testAccuracies) / 10
      
      
      