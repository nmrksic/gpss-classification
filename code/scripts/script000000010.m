i=10;likelihoodFunction=@likErf;covFunction={@covSum, { {@covMask, {[0  0  0  0  0  1], {@covSEiso}}},{@covProd, { {@covMask, {[0  0  0  0  1  0], {@covSEiso}}}, {@covMask, {[0  0  1  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  1  0  0], {@covSEiso}}} }}, {@covMask, {[1  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  1  0  0], {@covSEiso}}} }};hyperParameters.mean=0;hyperParameters.cov=[-0.34055    -0.58533      3.0763     -1.4644      2.6057     0.59356      2.4039      0.9367     0.80457    -0.69714      3.8043      1.7574];load('%(data_file)s');if(size(X,1)>250)subset=randsample(size(X,1),250);hyperParameters=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X(subset,:),y(subset));hypN=minimize(hyperParameters,@gp,-30,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);else;hypN=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);end;bicValue=gp(hypN,@infLaplace,@meanConst,covFunction,likelihoodFunction,X,y);save('%(output_file)s','bicValue','hypN');
