    hyp2.mean = 0.3328
    hyp2.cov = [3.0502 -1.2831]
    
    currId = 0;
    enc = [5];
    covF = encodeKernel(enc, 6); 

      trainAccs = zeros(1000, 1);
      testAccuracies = zeros(1000, 1);
      bicvals = zeros(1000, 1); 
    
    for seed = 0 :100:10000
        
        InitialiseRand(seed); 

         [trnX, trnY, tstX, tstY] = crossValidate(X, y, 10);

          for i = 1:10

              hyp = minimize(hyp2, @gp, -100, @infLaplace, @meanConst, covF, @likErf, trnX{i}, trnY{i});

              currId = currId + 1
              
              bicvals(currId) = gp(hyp, @infLaplace, @meanConst, covF, @likErf, trnX{i}, trnY{i});

              [~,~,~,~,lp2] = gp(hyp, @infLaplace, @meanConst, covF, @likErf, trnX{i}, trnY{i}, tstX{i}, ones(size(tstY{i})));
              testAccuracies(currId) = calculateAcc(lp2, tstX{i}, tstY{i});

              [~,~,~,~,lp1] =  gp(hyp, @infLaplace, @meanConst, covF, @likErf, trnX{i}, trnY{i}, trnX{i}, ones(size(trnY{i})));
              trainAccs(currId) = calculateAcc(lp1, trnX{i}, trnY{i});

          end
          
    end
    
    summary = [trainAccs, testAccuracies, bicvals]