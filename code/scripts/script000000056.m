i=56;likelihoodFunction=@likErf;covFunction= {@covMask, {[0  0  0  0  0  0  0  0  0  0  0  1  0], {@covSEiso}}};hyperParameters.mean=0;hyperParameters.cov=[0.18573      1.5925];load('%(data_file)s');if(size(X,1)>250)subset=randsample(size(X,1),250);hyperParameters=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X(subset,:),y(subset));hypN=minimize(hyperParameters,@gp,-30,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);else;hypN=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);end;bicValue=gp(hypN,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);save('%(output_file)s','bicValue','hypN');
