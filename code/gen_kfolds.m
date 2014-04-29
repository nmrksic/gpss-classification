function [trainfolds, testfolds] = gen_kfolds(X, y)
% Dave's code - not permuting data, as in run_one_fold:

N = size(y); 

randn('state', 0);
rand('twister', 0); 

perm = 1:N;

numFolds = 10;

if nargin < 3
    perm = 1:N;
end

ndx = 1;
for i=1:numFolds
  low(i) = ndx;
  Nbin(i) = fix(N/numFolds);
  if i==numFolds
    high(i) = N;
  else
    high(i) = low(i)+Nbin(i)-1;
  end
  testfolds{i} = low(i):high(i);
  trainfolds{i} = setdiff(1:N, testfolds{i});
  testfolds{i} = perm(testfolds{i});
  trainfolds{i} = perm(trainfolds{i});
  ndx = ndx+Nbin(i);
end


for i = 1:numFolds
    
    trainFoldsX{i} = X ( trainfolds{i}, : );
    trainFoldsY{i} = y( trainfolds );
    
    testFoldsX{i} = X ( testfolds{i}, : );
    testFoldsY{i} = y ( testfolds{i}, : );

end

