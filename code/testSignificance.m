function [significance, pvalues, averageErrors] = testSignificance(numMethods, numDatasets)

    significance = zeros(numMethods, numDatasets); % AIC, BIC, BIClight, Cross-validated, Biclight + likMix
    pvalues = zeros(numMethods, numDatasets);
    averageErrors = zeros(numMethods, numDatasets);
    filepaths = cell(numMethods, 1); % 5 different result folders
    expNames = cell(numDatasets,1); % 4 different data sets
    testAccs = cell(numMethods, 1); % for loading test accuracies of each method
    averageAccuracies = zeros(numMethods, 1);

    filepaths{1} = 'results/Archive/runs/AICruns/';
    filepaths{2} = 'results/Archive/runs/BICruns/';
    filepaths{3} = 'results/Archive/runs/BIClightruns/';
    filepaths{4} = 'results/Archive/runs/likmixruns/';
    filepaths{5} = 'results/Archive/runs/crossvalidationruns/';
    filepaths{6} = 'results/Archive/runs/randomforest/';

    expNames{1} = 'r_breast';
    expNames{2} = 'r_pima';
    expNames{3} = 'r_liver';
    expNames{4} = 'r_heart';


    % need to load the 10 test accuracies for all of them in each step.
    for names = 1 : numDatasets
        
        for method = 1 : numMethods

            loadpath = [filepaths{method}, expNames{names}, '/summary.mat'];
            load ( loadpath );
            testAccs{method} = testAccuracies;
            averageAccuracies(method) = averageAcc;
            averageErrors(method, names) = averageAcc;

        end
        % now actually evaluate the significance.
        [~, idx] = max(averageAccuracies) % find the best method's id

        significance ( idx, names ) = 1;
        pvalues(idx, names) = 1;

        for method = 1: numMethods
            if method ~= idx

                [~, p] = ttest(testAccs{idx}, testAccs{method})
                pvalues(method, names) = p;


                if p > 0.1
                    significance(method, names) = 1;
                end

            end
        end


    end
        
    averageErrors = 1 - averageErrors;

    averageErrors(5, 3) = nan;
    
end