i=149;covFunction={@covSum, {{@covProd, { {@covMask, {[0  0  0  0  0  0  1  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  1  0  0  0  0  0  0  0], {@covSEiso}}} }},{@covProd, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  0  0  0  0  0  1], {@covSEiso}}} }} }};hyperParameters.mean=0;hyperParameters.cov=[-0.13649     0.16031    0.028176      1.8058      0.5309     0.92525      2.0018     0.55929];load('%(data_file)s');if(size(X,1)>250)subset=randsample(size(X,1),250);hyperParameters=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,@likErf,X(subset,:),y(subset));hypN=minimize(hyperParameters,@gp,-30,@infLaplace,@meanConst,covFunction,@likErf,X,y);else;hypN=minimize(hyperParameters,@gp,-300,@infLaplace,@meanConst,covFunction,@likErf,X,y);end;bicValue=gp(hypN,@infLaplace,@meanConst,covFunction,@likErf,X,y);save('%(output_file)s','bicValue','hypN');