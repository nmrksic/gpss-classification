i=55;likelihoodFunction=@likErf;covFunction={@covProd, { {@covMask, {[0  1  0], {@covSEiso}}}, {@covMask, {[0  0  1], {@covSEiso}}} }};hyperParameters.mean=0;hyperParameters.cov=[-1.652      2.5926    -0.67214      1.3651];load('%(data_file)s');if(size(X,1)>250)subset=randsample(size(X,1),250);hyperParameters=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X(subset,:),y(subset));hypN=minimize(hyperParameters,@gp,-30,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);else;hypN=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);end;bicValue=gp(hypN,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);save('%(output_file)s','bicValue','hypN');
