function [] =  runExperiments(numExp, numDatasets)
% Starts the evaluation of the real world data sets... at this point, no
% heart, done manually. 

% Also, we are not trying feature selection.

    if (nargin < 1)
        numExp = 5;
    end

    if nargin < 2
        numDatasets = 6; % do all of the classification datasets.
    end

   % addpath ( genpath ( './' )); 

    system('mkdir -p outputs');
    system('mkdir -p data');
    system('mkdir -p scripts');
    system('mkdir -p images');

  %  system('rm scripts/*');

    expNames = cell(6, 1);

    expNames{1} = 'r_liver';
    expNames{2} = 'r_breast';
    expNames{3} = 'r_pima';
    expNames{4} = 'r_heart';
    expNames{5} = 'r_sonar';
    expNames{6} = 'r_ionosphere';

    for i = 1:numDatasets

        name = ['../Data/classification/', expNames{i}, '.mat']
        load(name);
        
        evaluateGPC(X, y, numExp, @infLaplace, 10, 1, expNames{i});

        if i == 4
             evaluateSyntheticData(5, 0, 1);  % after the four basic datasets, evaluate synthetic data as well. 
        end
        
    end
    

end


