i=41;likelihoodFunction=@likErf;covFunction={@covSum, {{@covProd, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  0  1  0  0  0  0], {@covSEiso}}} }}, {@covMask, {[0  1  0  0  0  0  0  0  0  0], {@covSEiso}}} }};hyperParameters.mean=0;hyperParameters.cov=[-0.090625     0.30787     0.45468       1.731    -0.13096   -0.053099];load('%(data_file)s');if(size(X,1)>250)subset=randsample(size(X,1),250);hyperParameters=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X(subset,:),y(subset));hypN=minimize(hyperParameters,@gp,-30,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);else;hypN=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);end;bicValue=gp(hypN,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);save('%(output_file)s','bicValue','hypN');
