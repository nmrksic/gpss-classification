
indices = crossvalind('Kfold',y,10)'

Xtr = X( ~ ( indices == 10  ) , :)

ytr = y( ~ ( indices == 10  ) , :)

Xtst = X(  ( indices == 10  ) , :)

ytst = y(  ( indices == 10  ) , :)

%[~, ~, ~, ~, trainAccs, kernelNames, bicValues, testAccuracies ] = AutomatedStatistician( Xtr, ytr, Xtst, ytst, 12 , 2, @infLaplace, 1, 0, 0)