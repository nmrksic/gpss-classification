    use_fixed_folds = 0;
    
    expNames = {'r_liver', 'r_breast', 'r_pima', 'r_heart'};
    
    for data_sets = 1:4
        
        if strcmp(expName, expNames(data_sets))
            use_fixed_folds = 1;
        end
        
    end
    
    if use_fixed_folds == 1
        for folds = 1:10
            name_to_load = ['dave_data/classification_', expName, '_fold_', num2str(folds), '_of_10.mat'];
            load(name_to_load)
            trnX{folds} = X;
            trnY{folds} = y;
            tstX{folds} = Xtest;
            tstY{folds} = ytest;
            
        end
    end