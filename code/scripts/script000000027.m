i=27;likelihoodFunction=@likErf;covFunction={@covSum, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  1  0  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  1  0  0  0  0  0  0  0  0], {@covSEiso}}} }} }};hyperParameters.mean=0;hyperParameters.cov=[-0.090625     0.30787    -0.13096   -0.053099      1.0912      1.4304];load('%(data_file)s');if(size(X,1)>250)subset=randsample(size(X,1),250);hyperParameters=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X(subset,:),y(subset));hypN=minimize(hyperParameters,@gp,-30,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);else;hypN=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);end;bicValue=gp(hypN,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);save('%(output_file)s','bicValue','hypN');
