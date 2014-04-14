function evaluateSyntheticData ( numExp, seed, runParallel)

    if nargin < 2
        seed=0;   
    end
    
    if nargin < 1
        numExp = 5;
    end
    
    if nargin < 3
        runParallel = 0; % don't run on fear by default.
    end

    InitialiseRand(seed);
    x_max = 2;

    results = cell(37, 7);
    
    inferenceMethod = @infLaplace;
    meanfunc = @meanConst;
    likfunc = @likErf;
    
    % only block to be modified in order to change generating kernel:
    covFunction{1} = {@covSum, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  0  1  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  0  0  1  0  0  0], {@covSEiso}}} }}, {@covMask, {[0  0  0  0  0  0  0  0  0  1], {@covSEiso}}} }};
    covParams{1} = [0 1 0 1 0 1 0 1];
    
    covFunction{2} = {@covSum, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  0  1  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  1  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  0  0  1  0  0  0], {@covSEiso}}} }}, {@covMask, {[0  0  0  0  0  0  0  0  1  0], {@covSEiso}}} }};
    covParams{2} = [-1 1 -1 1 -1 1 -1 1 -1 1];
    
    dims{1} = 10;
    dims{2} = 10;
    
    % Bayes optimal rate on each test data set in order to assess relative performance of the kernel constructed:
    %kernelOptimalRates = zeros ( length(covFunction) , 1);
    %bayesOptimalRates = zeros ( length(covFunction) , 1);

    experimentsSNR = [100, 1];
    experimentsSPnoise = [0, 0.05, 0.2]; 
    
    results{1,1} = 'N';     results{1,2} = 'SNR';     results{1,3} = 'sp\_noise'; 
        results{1,4} = 'Kernel Chosen';     results{1,5} = 'Test Accuracy';     
        results{1,6} = 'Kernel rate';     results{1,7} = 'Bayes optimal rate'; 
    
    n = 500; % we generate this data, then subsample it to get 100, 300 dataeva
    
    currentLine = 1; % line in results cell we write to (start at line 2). 
    
    fileprefix = 'results/SyntheticDataEvaluation/';

    system([' mkdir -p ', fileprefix]);
    
    fileprefix = [fileprefix, '/synthResults.mat'];
    
    format short; 
    
    for i = 1:length(covFunction)

        for SNR = experimentsSNR
            for salt_pepper_noise = experimentsSPnoise
                
                snris = SNR;
                spnoiseis = salt_pepper_noise;
                
                
                % draw data from GP prior
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

                maskRandom = rand(size(y));
                maskRandom = pullClasses(maskRandom, 1 - salt_pepper_noise)'; % there will be ~ outliersPercent indices with a value of 1. 
                countChanges = sum ( maskRandom == 1 );

                indicesToChange = find(maskRandom == 1);
                y( indicesToChange ) = pullClasses( rand( countChanges, 1), 0.5 ); 

                % create training and testing sets
                X_trn = X(1:n, : );  
                y_trn = y(1:n);
                X_tst = X((n+1):(2*n), :); 
                y_tst = y( (n+1): (2*n) );
                y_tstPreNoiseOriginal = yNoiseLess( (n+1): (2*n) );

                % Calculate kernel and Bayes optimal rates
                bayesOptimalRates = calculateAcc(y_tstPreNoiseOriginal, X_tst, y_tst); % Same for all three data set sizes

                for sample = [500, 300, 100]
                    
                    if ( sample == 300) % we need to subsample
                    
                        sampler = randsample(1:500, sample);
                        X_trn = X_trn(sampler, :);
                        y_trn = y_trn(sampler, :);
                        X_tst = X_tst(sampler, :);
                        y_tst = y_tst(sampler, :);

                    elseif sample == 100
                        
                        sampler = randsample(1:300, sample);
                        X_trn = X_trn(sampler, :);
                        y_trn = y_trn(sampler, :); 
                        X_tst = X_tst(sampler, :); 
                        y_tst = y_tst(sampler, :); 
                    
                    end
                
                    trueHyperparams.cov = covParams{i};
                    [~,~,~,~,lp] = gp(trueHyperparams, inferenceMethod, [], covFunction{i}, likfunc, X_trn, y_trn, X_tst, ones(size(y_tst)));
                    kernelOptimalRates = calculateAcc(lp, X_tst, y_tst);

                    % Then do structure discovery:

                   [~, ~, ~, ~, ~, ~, ~ , testAccuracies, ~, finalEncoder ] = ...
                       AutomatedStatistician(X_trn, y_trn, X_tst, y_tst, 4 * dims{i} , numExp, runParallel, inferenceMethod, 0, 0);
                    
                    currentLine = currentLine + 1;
                    
                    disp(['Evaluated ', num2str(currentLine - 1), ' out of 36 experiments.']);
                   
                    results{currentLine, 1} = [num2str(sample), '& ' ];
                    results{currentLine, 2} =  [num2str(SNR), '& ' ];
                    results{currentLine, 3} = [num2str(round(salt_pepper_noise*100)), '\% & '];
                    results{currentLine, 4} = ['$', decodeKernelToLatex( finalEncoder ), '$ &' ]; 
                    results{currentLine, 5} = [ num2str(round( testAccuracies * 10000) / 100 ), '\% & '];
                    results{currentLine, 6} = [ num2str(round( kernelOptimalRates * 10000) / 100), '\% & '];
                    
                    if sample == 500
                        results{currentLine, 7} =  [num2str(round( bayesOptimalRates * 10000 ) /  100), '\%  \\ \hline '];
                    else
                        results{currentLine, 7} =  [num2str(round( bayesOptimalRates * 10000 ) /  100), '\%  \\  '];
                    end
                    
               
                    fileprefixAux = 'results/SyntheticDataEvaluation/synthResultsTemp.mat';
                    save(fileprefixAux, 'results');
                
                end
            end 
        end
    end

  
    save(fileprefix, 'results');
  
    results2  = cell( 37, 7 );
    for i = 2:3:37
        for j = 1:7
            results2{i+2, j} = results{i, j};
        end
    end
    
    for i = 4:3:37
        for j = 1:7
            results2{i-2, j} = results{i, j};
        end
    end
    
    
    
end
