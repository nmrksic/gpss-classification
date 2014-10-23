function [] = BayesianModelAveraging(expName, runParallel)

    if nargin < 2
        runParallel = 0;
    end

    load(['../Data/classification/', expName, '.mat']);
    
    covFunctions = cell(10, 1);
    Scores = cell(10, 1);
    Hypers = cell(10, 1);

    seed = 0;
    InitialiseRand(seed); % random seed initialised here. 

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, 10); % numFolds-fold cross validation data sets

    useDaveFolds % Comment this if you don't want to use Dave's folds
    
    filepath = 'results/Archive/runs/BIClightruns/';
    
   
    for folds = 1:10 
        
        disp(['evaluating fold ', num2str(folds)]);
        
        filePathNew = [filepath, expName, '/fold', num2str(folds), '/searchStats.mat'];
        
        load(filePathNew); 
        
        BicValsList(BicValsList==0) = [];
            
        [~, id] = min(BicValsList); % finds the index of the best kernel
        
        % we now expand the previous one and this one to get all values. 

        [~, ~, ~, allCovFuns1, allScores1, allHypers1] = nextKernel(trnX{folds}, trnY{folds}, squeeze( allEncoderMatrices(id-1, :, :)), hyperList{id-1}, 1, runParallel, @infLaplace, @likErf, 0);
                disp('tag1')

        [~, ~, ~, allCovFuns2, allScores2, allHypers2] = nextKernel(trnX{folds}, trnY{folds}, squeeze( allEncoderMatrices(id, :, :)), hyperList{id}, 1, runParallel, @infLaplace, @likErf, 0);
                disp('tag2')

        [~, ~, ~, allCovFuns3, allScores3, allHypers3] = nextKernel(trnX{folds}, trnY{folds}, squeeze( allEncoderMatrices(id-2, :, :)), hyperList{id-2}, 1, runParallel, @infLaplace, @likErf, 0);
                disp('tag3')


        allCovFuns = [allCovFuns1; allCovFuns2; allCovFuns3];
        
        allScores = [allScores1; allScores2; allScores3];
        
        allHypers = [allHypers1; allHypers2; allHypers3];
        
        covFunctions{folds} = allCovFuns;
        Scores{folds} = allScores;
        Hypers{folds} = allHypers;
        
    end
          
   
    save(['results/BMA/', expName], 'covFunctions', 'Scores', 'Hypers'); 
    
end