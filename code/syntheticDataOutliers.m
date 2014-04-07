function syntheticDataOutliers (n, SNR, outliersPercent, seed)

    if nargin < 4
        seed=0;   
    end

    InitialiseRand(seed);
    x_max = 2;

    data_folder = 'data/syntheticDataOutliers/';
    system( 'mkdir -p data');
    system('mkdir -p data/syntheticDataOutliers');  % initialise data directory in current folder. 

    covFunction{1} = {@covSum, { {@covMask, {[1  0  0], {@covSEiso}}}, {@covMask, {[0  1  0], {@covSEiso}}}, {@covMask, {[0  0  1], {@covSEiso}}} }};

    covParams{1} = [0 2 -1 2 1 2];
    dims{1} = 3;
    names{1} = 'SE1_plus_SE2_plus_SE3.mat';

    for i = 1:length(covFunction)

        X = (rand(2*n,dims{i})-0.5)*2*x_max;
        K = feval(covFunction{i}{:}, covParams{i}, X);
        K = K + 1e-4*eye(2*n);
        y = chol(K)' * randn(2*n,1);
        y = y - mean(y); % 0-center
        y = y / std(y);  % normalise
        y = y + randn(size(y)) * (1/SNR^0.5);  
        y = pullClasses(y, 0);

        y = y';
         
        maskRandom = rand(size(y));
        
        maskRandom = pullClasses(maskRandom, 1 - outliersPercent)'; % there will be ~ outliersPercent indices with a value of 1. 
        
        countChanges = sum ( maskRandom == 1 );
        
        y( maskRandom == 1 ) = pullClasses( rand( countChanges, 1) ); 

        X_trn = X(1:n, : );  
        y_trn = y(1:n);
        X_tst = X((n+1):(2*n), :); 
        y_tst = y( (n+1): (2*n) );


        save([data_folder names{i}], 'X_trn', 'y_trn', 'X_tst', 'y_tst');
    
        % Then do structure discovery:
   
        [~, ~, ~, ~, trainAccs, kernelNames, bicValues, testAccuracies ] = ...
                 AutomatedStatistician(X_trn, y_trn, X_tst, y_tst, 2 * dims{i} , 5, 0, @infLaplace, 0, 0); % do not run in parallel (i.e. run locally)
    
    end
    
end

% feed noise-to signal as 0.01 1 and 100

% 1, 10, 100 to be tried.