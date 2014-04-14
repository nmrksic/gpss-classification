function [  ] = crossValidateSave(expname)

load(['../Data/classification/', expname, '.mat']);

seed = 0;
InitialiseRand(seed);

numFolds = 10;
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

%% save separate folds

for i = 1: numFolds
    X = trainFoldsX{i}; y = trainFoldsY{i};
    Xtest = testFoldsX{i}; ytest = testFoldsY{i};
    system(['mkdir -p data/', expname]);
    save(['data/', expname, '/' expname, '_fold_', num2str(i), '_of_' , num2str(numFolds)], 'X', 'y', 'Xtest', 'ytest');
end
%% end of code