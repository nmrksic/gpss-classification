 load('%(data_file)s');
 
 covFunction = covFunctions{i};
 hyperParameters = finalHyperParams{i}; 
 
 inferenceMethod = @infLaplace;
 likfunc = @likErf;
 meanfunc = @meanConst; 

 hypN = minimize(hyperParameters, @gp, -300, inferenceMethod, meanfunc, covFunction, likfunc, X, y);
 nlml = gp(hypN, inferenceMethod, meanfunc, covFunction, likfunc, X, y);

 sampleSize = size(X, 1); 
 
 bicValue = nlml; 
 
 save ('%(output_file)s', 'bicValue', 'hypN');
 
 
