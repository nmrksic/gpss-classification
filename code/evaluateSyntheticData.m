function evaluateSyntheticData ( numRestarts, seed, runParallel, likelihoodMod)

    if nargin < 2
        seed=0;   
    end
    
    if nargin < 1
        numRestarts = 5;
    end
    
    if nargin < 3
        runParallel = 0; % don't run on fear by default.
    end
    
    if nargin < 4
        likelihoodMod = 0; % 0 is @likErf, 1 is likMix
    end
    
    if likelihoodMod == 0
        likelihoodFunction = @likErf;
    elseif likelihoodMod == 1
        likelihoodFunction = {@likMix,{@likUni,@likErf}};
    end

    InitialiseRand(seed);
    x_max = 2;

    results = cell(73, 8);
    
    inferenceMethod = @infLaplace;
    meanfunc = @meanConst;
    searchCriterion = 0; % prove that this works the same with BIC first.
    
    % only block to be modified in order to change generating kernel:
   
    covFunction{1}  =  {@covSum, { {@covMask, {[1  0  0], {@covSEiso}}}, {@covMask, {[0  1  0], {@covSEiso}}}, {@covMask, {[0  0  1], {@covSEiso}}} }};
    covParams{1} = [0 1 0 1 0 1];
    
    covFunction{2}  =   {@covSum, { {@covMask, {[1  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  1  0  0], {@covSEiso}}}, {@covMask, {[0  0  1  0], {@covSEiso}}} }}, {@covMask, {[0  0  0  1], {@covSEiso}}} }};
    covParams{2} = [0 1 0 1 0 1 0 1];

    covFunction{3} = {@covSum, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  0  1  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  0  0  1  0  0  0], {@covSEiso}}} }}, {@covMask, {[0  0  0  0  0  0  0  0  0  1], {@covSEiso}}} }};
    covParams{3} = [0 1 0 1 0 1 0 1];
    
    covFunction{4} = {@covSum, { {@covMask, {[1  0  0  0  0  0  0  0  0  0], {@covSEiso}}},{@covProd, { {@covMask, {[0  0  1  0  0  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  1  0  0  0  0  0], {@covSEiso}}}, {@covMask, {[0  0  0  0  0  0  1  0  0  0], {@covSEiso}}} }}, {@covMask, {[0  0  0  0  0  0  0  0  1  0], {@covSEiso}}} }};
    covParams{4} = [0 1 0 1 0 1 0 1 0 1];
    
    dims{1} = 3;
    dims{2} = 4;
    dims{3} = 10;
    dims{4} = 10;
    
        
    % Bayes optimal rate on each test data set in order to assess relative performance of the kernel constructed:
    %kernelOptimalRates = zeros ( length(covFunction) , 1);
    %bayesOptimalRates = zeros ( length(covFunction) , 1);

    experimentsSNR = [100, 1];
    experimentsSPnoise = [0, 0.05, 0.2]; 
    
    results{1,1} = 'N';     results{1,2} = 'SNR';     results{1,3} = 'sp\_noise'; 
        results{1,4} = 'Kernel Chosen';     results{1,5} = 'Test Accuracy';     
        results{1,6} = 'Kernel rate';     results{1,7} = 'Bayes optimal rate'; 
    
    n = 500; % we generate this data, then subsample it to get 100, 300 data points.
    
    currentLine = 1; % line in results cell we write to (start at line 2). 
    
    fileprefix = 'results/SyntheticDataEvaluation/';

    system([' mkdir -p ', fileprefix]);
    
    fileprefix = [fileprefix, 'synthResults'];
    
    format short; 
    
     
    sampler1 = randsample(1:500, 300);
    sampler2 = randsample(1:300, 100);

    maskRandom = rand(1, 1000); % only generate once, to control variability between different experiments. 

    for i = 1:length(covFunction)

        % draw data from GP prior
        X = (rand(2*n,dims{i})-0.5)*2*x_max;
        K = feval(covFunction{i}{:}, covParams{i}, X);
        K = K + 1e-4*eye(2*n);
        y = chol(K)' * randn(2*n,1);
        y = y - mean(y); % 0-center
        y = y / std(y);  % normalise
        yNoiseLess = pullClasses(y, 0); % for determining the Bayes Optimal rate
        
        for SNR = experimentsSNR
            
              yn = y + randn(size(y)) * (1/SNR^0.5);   % adding white noise
              yn = pullClasses(yn, 0);
              yn = yn';
            
            
            for salt_pepper_noise = experimentsSPnoise
                
                   
                maskRandomNew = pullClasses(maskRandom, 1 - salt_pepper_noise)'; % there will be ~ outliersPercent indices with a value of 1. 
                countChanges = sum ( maskRandomNew == 1 );
                indicesToChange = find(maskRandomNew == 1);
                
                for change = 1:size(indicesToChange)
                
                    yn( indicesToChange(change) ) =  - yn(indicesToChange(change)); % flip those. 
                    
                end

                % create training and testing sets
                X_trn = X(1:n, : );  
                y_trn = yn(1:n);
                X_tst = X((n+1):(2*n), :); 
                y_tst = yn( (n+1): (2*n) );
                y_tstPreNoiseOriginal = yNoiseLess( (n+1): (2*n) );

                % Calculate kernel and Bayes optimal rates
                bayesOptimalRates = calculateAcc(y_tstPreNoiseOriginal, X_tst, y_tst)   % Same for all three data set sizes

                for sample = [500, 300, 100]
                    
                    if ( sample == 300) % we need to subsample
                    
                        X_trn = X_trn(sampler1, :);
                        y_trn = y_trn(sampler1, :);
                        X_tst = X_tst(sampler1, :);
                        y_tst = y_tst(sampler1, :);

                    elseif sample == 100
                        
                        X_trn = X_trn(sampler2, :);
                        y_trn = y_trn(sampler2, :); 
                        X_tst = X_tst(sampler2, :); 
                        y_tst = y_tst(sampler2, :); 
                    
                    end
                
                    trueHyperparams.cov = covParams{i};
                    
                    if likelihoodMod == 1
                        trueHyperparams.lik = [-1 1]; % set this to a fixed value, in order to always get the same final result. 
                    end
                    
                    [~,~,~,~,lp] = gp(trueHyperparams, inferenceMethod, [], covFunction{i}, likelihoodFunction, X_trn, y_trn, X_tst, ones(size(y_tst)));
                    kernelOptimalRates = calculateAcc(lp, X_tst, y_tst)

                    % Then do structure discovery:

                   [~, ~, ~, ~, ~, ~, ~ , testAccuracy, finalHyperParam, finalEncoder ] = ...
                       structureSearch(X_trn, y_trn, X_tst, y_tst, 4 * dims{i}, numRestarts, runParallel, inferenceMethod, likelihoodFunction, searchCriterion);
                    
                    currentLine = currentLine + 1;
                    
                    disp(['Evaluated ', num2str(currentLine - 1), ' out of 72 experiments.']);
                   
                    results{currentLine, 1} = [num2str(sample), '& ' ];
                    results{currentLine, 2} =  [num2str(SNR), '& ' ];
                    results{currentLine, 3} = [num2str(round(salt_pepper_noise*100)), '\% & '];
                    results{currentLine, 4} = ['$', decodeKernelToLatex( finalEncoder ), '$ &' ]; 
                    results{currentLine, 5} = [ num2str(round( testAccuracy * 10000) / 100 ), '\% & '];
                    results{currentLine, 6} = [ num2str(round( kernelOptimalRates * 10000) / 100), '\% & '];
                    
                    if sample == 500
                        results{currentLine, 7} =  [num2str(round( bayesOptimalRates * 10000 ) /  100), '\%    '];
                    else
                        results{currentLine, 7} =  [num2str(round( bayesOptimalRates * 10000 ) /  100), '\%    '];
                    end
                    
                    if likelihoodMod == 1
                        
                        likhyper = finalHyperParam.lik;
                        ratio = exp( likhyper(1) ) / ( exp(likhyper(1)) + exp(likhyper(2)) ); % get it as percent 
                        results{currentLine, 8} = [' &  ', num2str( round( ratio * 10000 ) /  100), '\%  \\ '];
                    
                    else
                        results{currentLine, 8} = ' ';
                    end
               
                    fileprefixAux = ['results/SyntheticDataEvaluation/synthResultsTemp', num2str(likelihoodMod) , '.mat'];
                    save(fileprefixAux, 'results');
                
                end
            end 
        end
    end

  
    results2  = cell( 73, 8 );
    
    for i = 2:3:73
        for j = 1:8
            results2{i+2, j} = results{i, j};
        end
    end
    
    for i = 4:3:73
        for j = 1:8
            results2{i-2, j} = results{i, j};
        end
    end
    for i = 3:3:73
        for j = 1:8
            results2{i, j} = results{i, j};
        end
    end
    
    save( [ fileprefix, num2str(likelihoodMod), '.mat' ], 'results', 'results2');
    
end
