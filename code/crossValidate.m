function [ trainFoldsX, trainFoldsY, testFoldsX, testFoldsY ] = crossValidate(X, y, numFolds)

% Dave's folds:
% N = size(X, 1); 
% 
% randn('state', 0);
% rand('twister', 0); 
% 
% perm = 1:N;
% if nargin < 3
%     numFolds = 10;
% end
% 
% ndx = 1;
% for i=1:numFolds
%   low(i) = ndx;
%   Nbin(i) = fix(N/numFolds);
%   if i==numFolds
%     high(i) = N;
%   else
%     high(i) = low(i)+Nbin(i)-1;
%   end
%   testfolds{i} = low(i):high(i);
%   trainfolds{i} = setdiff(1:N, testfolds{i});
%   testfolds{i} = perm(testfolds{i});
%   trainfolds{i} = perm(trainfolds{i});
%   ndx = ndx+Nbin(i);
% end
% 
% 
% for i = 1:numFolds
%     
%     trainFoldsX{i} = X ( trainfolds{i}, : );
%     trainFoldsY{i} = y( trainfolds{i} );
%     
%     testFoldsX{i} = X ( testfolds{i}, : );
%     testFoldsY{i} = y ( testfolds{i}, : );
% 
% end
% 

% my folds, code: 

indices = crossvalind('Kfold',y,numFolds);
test = cell(numFolds);
train = cell(numFolds);

for i = 1:numFolds
    
    test{i} = (indices == i);
    train{i} = ~test{i};
    trainFoldsX{i} = X ( train{i}, : );
    trainFoldsY{i} = y( train{i}, : );
    
    testFoldsX{i} = X ( test{i}, : );
    testFoldsY{i} = y ( test{i}, : );

end



end