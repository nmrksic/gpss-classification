function [significance, pvalues] = testSignificance(numMethods, numDatasets)

significance = zeros(numMethods, numDatasets); % AIC, BIC, BIClight, Cross-validated, Biclight + likMix
pvalues = zeros(numMethods, numDatasets);

filepaths = cell(numMethods, 1); % 5 different result folders
expNames = cell(numDatasets,1); % 4 different data sets
testAccs = cell(numMethods, 1); % for loading test accuracies of each method
averageAccuracies = zeros(numMethods, 1);

filepaths{1} = 'results/AICruns/';
filepaths{2} = 'results/BICruns/';
filepaths{3} = 'results/BIClightruns/';
filepaths{4} = 'results/runs/likmixruns/';
filepaths{5} = 'results/runs/crossvalidationruns/';

expNames{1} = 'r_liver';
expNames{2} = 'r_pima';
expNames{3} = 'r_heart';
expNames{4} = 'r_breast';


% need to load the 10 test accuracies for all of them in each step.
for names = 1 : numDatasets
    for method = 1 : numMethods
        
        loadpath = [filepaths{method}, expNames{names}, '/summary.mat'];
        load ( loadpath );
        testAccs{method} = testAccuracies;
        averageAccuracies(method) = averageAcc; 
        
    end
   % now actually evaluate the significance.  
   [~, idx] = max(averageAccuracies); % find the best method's id
   
   significance ( idx, names ) = 1;
   pvalues(idx, names) = 1;
   
   for method = 1: numMethods
      if method ~= idx
          
          [~, p] = ttest(testAccs{idx}, testAccs{method});
          pvalues(method, names) = p;
          
          
          if p > 0.975
              significance(method, names) = 1;
          end
          
      end
   end
   
   
end
        

