function syntheticDataOutliers (n, SNR, salt_pepper_noise, numExp, seed, runParallel)

    if nargin < 5
        seed=0;   
    end
    
    if nargin < 4
        numExp = 5;
    end
    
    if nargin < 6
        runParallel = 0; % don't run on fear by default.
    end

    InitialiseRand(seed);
    x_max = 2;

    data_folder = 'data/syntheticDataOutliers/';
    system( 'mkdir -p data');
    system('mkdir -p data/syntheticDataOutliers');  % initialise data directory in current folder. 

    inferenceMethod = @infLaplace;
    meanfunc = @meanConst;
    likfunc = @likErf;
    
    % only block to be modifid in order to change generating kernel:
    covFunction{1} = {@covSum, { {@covMask, {[1  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  1  0  0], {@covSEiso}}}, {@covMask, {[0  0  1  0], {@covSEiso}}} }}, {@covMask, {[0  0  0  1], {@covSEiso}}} }};
    covParams{1} = [0 1 0 1 0 1 0 1];
    
    covFunction{2} = {@covSum, { {@covMask, {[1  0  0], {@covSEiso}}}, {@covMask, {[0  1  0], {@covSEiso}}}, {@covMask, {[0  0  1], {@covSEiso}}} }};
    covParams{2} = [-1 2 0 2 1 2];
    
    dims{1} = 4;
    names = 'SE1+SE2XSE3+SE4';
    % ....
    
    % Bayes optimal rate on each test data set in order to assess relative performance of the kernel constructed:
    %kernelOptimalRates = zeros ( length(covFunction) , 1);
    %bayesOptimalRates = zeros ( length(covFunction) , 1);
    
    for i = 1:length(covFunction)

        X = (rand(2*n,dims{i})-0.5)*2*x_max;
        K = feval(covFunction{i}{:}, covParams{i}, X);
        K = K + 1e-4*eye(2*n);
        y = chol(K)' * randn(2*n,1);
        y = y - mean(y); % 0-center
        y = y / std(y);  % normalise
        yNoiseLess = pullClasses(y, 0); % for determining the Bayes Optimal rate
        y = y + randn(size(y)) * (1/SNR^0.5);   % adding white noise
        y = pullClasses(y, 0);

        y = y';
        
        yold = y;
        
        maskRandom = rand(size(y));
        
        maskRandom = pullClasses(maskRandom, 1 - salt_pepper_noise)'; % there will be ~ outliersPercent indices with a value of 1. 
        
        countChanges = sum ( maskRandom == 1 );
        
        indicesToChange = find(maskRandom == 1);
                
        y( indicesToChange ) = pullClasses( rand( countChanges, 1), 0.5 ); 
        
   
        X_trn = X(1:n, : );  
        y_trn = y(1:n);
        X_tst = X((n+1):(2*n), :); 
        y_tst = y( (n+1): (2*n) );

        y_tstPreNoise = yNoiseLess( (n+1): (2*n) );
        
        trueHyperparams.cov = covParams{i};
        [~,~,~,~,lp] = gp(trueHyperparams, inferenceMethod, [], covFunction{i}, likfunc, X_trn, y_trn, X_tst, ones(size(y_tst)));
        kernelOptimalRates = calculateAcc(lp, X_tst, y_tst);
       
        bayesOptimalRates = calculateAcc(y_tstPreNoise, X_tst, y_tst); 
        
        msg1 = ['Kernel Optimal rate: ', num2str(kernelOptimalRates)];
        msg2 = ['Bayes (function) Optimal rate: ', num2str(bayesOptimalRates)];
        
        
         % ------------to remove---------------
         disp(msg1);
         disp(msg2);
        
         disp(' ');
         %return;
         % ---------------------------------
        
        % save([data_folder names{i}], 'X_trn', 'y_trn', 'X_tst', 'y_tst');
    
        % Then start structure discovery:
   
        fileprefix = ['"', 'results/' ,names, '; N:', num2str(n) , '; SNR: ', num2str(SNR), '; spnoise: ', num2str(salt_pepper_noise), '"' ,'/'];     
        
        system([' mkdir -p ', fileprefix]);

        fileprefix = ['results/' ,names, '; N:', num2str(n) , '; SNR: ', num2str(SNR), '; spnoise: ', num2str(salt_pepper_noise), '/'];     
        
        %fileprefix = ['/', fileprefix];
                
        % [kernelNamesList, BicValsList, testAccuracciesList, hyperList, trainAccuraciesList, kernelNames, bicValues, testAccuracies, bestHypers, finalEncoder ] = ...
        %         AutomatedStatistician(X_trn, y_trn, X_tst, y_tst, 3 * dims{i} , numExp, runParallel, inferenceMethod, 0, 0); 
    
       
        disp(' ');
        disp('Running the structure search...');
        disp(' ');
        
        kernelSearchLog = evalc('[kernelNamesList, BicValsList, testAccuracciesList, hyperList, trainAccuraciesList, kernelNames, bicValues, testAccuracies, bestHypers, finalEncoder ] = AutomatedStatistician(X_trn, y_trn, X_tst, y_tst, 3 * dims{i} , numExp, runParallel, inferenceMethod, 0, 0);');
        %[kernelNamesList, BicValsList, testAccuracciesList, hyperList, trainAccuraciesList, kernelNames, bicValues, testAccuracies, bestHypers, finalEncoder ] = AutomatedStatistician(X_trn, y_trn, X_tst, y_tst, 3 * dims{i} , numExp, runParallel, inferenceMethod, 0, 0);
        % plotPosteriors(X_trn, y_trn, finalEncoder, bestHypers, fileprefix);
        
        kernelNamesList = kernelNamesList';
        BicValsList = BicValsList';
        testAccuracciesList =testAccuracciesList';
        hyperList = hyperList';
        trainAccuraciesList = trainAccuraciesList'; % for the sake of saving them nicely - could be moved to AutomatedStatistician.m
        
        save ( [ fileprefix, 'searchStats.mat'], 'kernelNamesList', 'BicValsList', 'trainAccuraciesList', ...
            'testAccuracciesList', 'hyperList', 'kernelOptimalRates', 'bayesOptimalRates', 'n', 'SNR', 'salt_pepper_noise', 'numExp', 'seed', 'names', 'covFunction', 'covParams' );
        
        fileID = fopen([fileprefix, 'kernelSearchLog.txt'], 'w');
        fprintf(fileID, '%s\n', msg1);
        fprintf(fileID, '%s\n', msg2);
        fprintf(fileID, '%s\n', ' ');
        fprintf(fileID, '%s\n', kernelSearchLog);
        fclose(fileID);
        
        % This code, with set seeds, runs the same on
        % fear and locally - this has been confirmed, AutomatedStatistician 
        % returns the same figures when run on fear or laplace (minor numeric instability). 
        
        
        
    end
    
    
    
end
