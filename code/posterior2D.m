
hyp.mean = -1.0413;
hyp.cov = [3.7268 0.1000 3.1659 0.0973 2.0124 -1.0172 -0.0752 0.6764];

encoder = [2 0; 8 0; 6 7];

covFunction = encodeKernel(encoder, 8); 

dimNames{1} = 'Number of times pregnant';
dimNames{2} = 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test';
dimNames{3} = 'Diastolic blood pressure (mm Hg)';
dimNames{4} = 'Triceps skin fold thickness (mm)';
dimNames{5} = '2-Hour serum insulin (mu U/ml)';
dimNames{6} = 'Body mass index (weight in kg per (height in m) squared)';
dimNames{7} = 'Diabetes pedigree function';
dimNames{8} = 'Age (years)';

plotPosteriors(X, y, encoder, hyp, 'pimarun', dimNames);

