[successfulEval] =  runExperiments(numExp)
% Starts the evaluation of the real world data sets... at this point, no
% heart, done manually. 

%Also, we are not trying feature selection.

    if (nargin < 1)
        numExp = 5;
    end

    addpath ( genpath ( './' )); 

    system('mkdir -p outputs');
    system('mkdir -p data');
    system('mkdir -p scripts');
    system('mkdir -p images');

    system('rm scripts/*');

    expNames = cell(6, 1);

    expNames{1} = 'r_liver';
    expNames{2} = 'r_breast';
    expNames{3} = 'r_pima';
    expNames{4} = 'r_ionosphere';
    expNames{5} = 'r_sonar';
    expNames{6} = 'r_heart';

    evaluateExperiment(expNames, 5, 5); % up to sonar, 5 restarts...

    for i = 1:numDatasets

        name = ['../Data/classification/', expName{i}, '.mat'];
        load(name);

        evaluateGPC(X, y, numExp, @infLaplace, 10, 1, exp_name{1});

    end

    successfulEval = 1389;