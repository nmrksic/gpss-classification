function [ trainFoldsX, trainFoldsY, testFoldsX, testFoldsY ] = crossValidate(X, y, numFolds)

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