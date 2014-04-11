
if strcmp(expName, 'r_pima')
    
    dimensionLabels = cell(8, 1);
    
    dimensionLabels{1} = 'Number of times pregnant';
    dimensionLabels{2} = 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test';
    dimensionLabels{3} = 'Diastolic blood pressure (mm Hg)';
    dimensionLabels{4} = 'Triceps skin fold thickness (mm)';
    dimensionLabels{5} = '2-Hour serum insulin (mu U/ml)';
    dimensionLabels{6} = 'Body mass index (weight in kg per (height in m) squared)';
    dimensionLabels{7} = 'Diabetes pedigree function';
    dimensionLabels{8} = 'Age (years)'; 
    fprintf('Starting the pima experiment. ');

end


if strcmp(expName, 'r_liver')
    
    dimensionLabels = cell(6, 1);

    
    dimensionLabels{1} = 'Mcv mean corpuscular volume';
    dimensionLabels{2} = 'Alkphos alkaline phosphotase';
    dimensionLabels{3} = 'Sgpt alamine aminotransferase'; 
    dimensionLabels{4} = 'Sgot aspartate aminotransferase';
    dimensionLabels{5} = 'Gammagt gamma-glutamyl transpeptidase';
    dimensionLabels{6} = 'Drinks number of half-pint equivalents of alcoholic beverages drunk per day';
    fprintf('Starting the liver experiment. ');

end

if strcmp(expName, 'r_heart')
        dimensionLabels = cell(13, 1);
    dimensionLabels{1} = 'Age (years)';
    dimensionLabels{2} = 'Sex (1 male, 0 female)';
    dimensionLabels{3} = 'Chest pain type: typical angina, atypical angina, non-anginal pain, asymptomatic.';
    dimensionLabels{4} = 'Resting blood pressure (in mm Hg on admission to the hospital).';
    dimensionLabels{5} = 'Serum cholestoral in mg/dl';
    dimensionLabels{6} = 'Fasting blood sugar > 120 mg/dl: (1 = true; 0 = false)';
    dimensionLabels{7} = 'Resting electrocardiographic results: normal, ST-T wave abnormality, ventricular hypertrophy';
    dimensionLabels{8} = 'Maximum heart rate achieved'; 
    dimensionLabels{9} = 'Exercise induced angina (1 = yes; 0 = no)';
    dimensionLabels{10} = 'ST depression induced by exercise relative to rest.';
    dimensionLabels{11} = 'Slope of the peak exercise ST segment: upsloping, flat, downsloping ';
    dimensionLabels{12} = 'Number of major vessels (0-3) colored by flourosopy.'; 
    dimensionLabels{13} = 'Thalassemias (blood disorder): 3 = normal, 6 = fixed defect, 7 = reversable defect';
    fprintf('Starting the heart experiment. ');

end

if strcmp(expName, 'r_breast')
        dimensionLabels = cell(9, 1);
    dimensionLabels{1} = 'Clump Thickness';
    dimensionLabels{2} = 'Uniformity of Cell Size';
    dimensionLabels{3} = 'Uniformity of Cell Shape';
    dimensionLabels{4} = 'Marginal Adhesion';
    dimensionLabels{5} = 'Single Epithelial Cell Size';
    dimensionLabels{6} = 'Bare Nuclei';
    dimensionLabels{7} = 'Bland Chromatin';
    dimensionLabels{8} = 'Normal Nucleoli'; 
    dimensionLabels{9} = 'Mitoses';
    fprintf('Starting the breast experiment. ');

end