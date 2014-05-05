function evaluateSyntheticNoSP ( numRestarts, runParallel)

    seed = 0;
    InitialiseRand(seed);

    if nargin < 1
        numRestarts = 1;
    end
    
    if nargin < 2
        runParallel = 0; % don't run on fear by default.
   end
    

    likelihoodFunction = @likErf;
    inferenceMethod = @infLaplace;
    searchCriterion = 0; % use BIC (light) as search criterion.

  %  format long
    
    experimentCount = 1;
    encoders = cell(10, 1);
    dims = cell(10, 1); 
    covFunction = cell(experimentCount, 1);
    covParams = cell(experimentCount, 1); 
        
    encoders{7} = [1 0 0; 0 0 0; 0 0 0];            dims{7} = 3;     % SE1 in 3D
    encoders{2} = [2 0 0; 2 0 0; 2 0 0];            dims{2} = 3;     % SE2 + SE2 + SE2, 3D
    encoders{3} = [2 3 0; 0 0 0; 0 0 0];            dims{3} = 4;     % SE2 x SE3, 4D
    encoders{4} = [2 3 0; 1 0 0; 4 0 0];            dims{4} = 4;     % SE1 + SE2 x SE3 + SE4, 4D
    encoders{5} = [2 3 0; 1 0 0; 4 0 0];            dims{5} = 10;    % SE1 + SE2 x SE3 + SE4, 10D
    encoders{6} = [2 3 0; 1 0 0; 4 0 0; 5 6 0];     dims{6} = 10;    % SE1 + SE2 x SE3 + SE4 + SE5 X SE6, 10D
    encoders{1} = [3 5 7; 0 0 0; 0 0 0];            dims{1} = 10;    % SE3 X SE5 X SE7, 10D
    encoders{8} = [3 5 7; 1 0 0; 10 0 0];           dims{8} = 10;   % SE1 + SE3 X SE5 X SE7 + SE10, 10D
    encoders{9} = [3 5 7 9; 0 0 0 0; 0 0 0 0];      dims{9} = 10;   % SE3 X SE5 X SE7 X SE9, 10D
    encoders{10} = [3 5 7 9; 1 0 0 0; 10 0 0 0];    dims{10} = 10;   % SE1 + SE3 X SE5 X SE7 X SE9 + SE10, 10D
 
    bayesValues = zeros(experimentCount, 2); 
    
    for i = 1:experimentCount
        
        covFunction{i} = encodeKernel(encoders{i}, dims{i});
        numHyper = nnz(encoders{i}) * 2;
        temp = zeros(numHyper, 1); % all lengthscales set to 1, that is 0 in log-domain
        covParams{i} = temp;
       
    end
    
    % modify lengthscales for those kernels with repeating dimensions (2, 3, 4)
   % covParams{2} = [1 0 0 0 -1 0];
 %   covParams{3} = [1 0 0 0];
 %   covParams{2} = [1 0 0 0 ];

    results = cell(experimentCount * 3, 9);
    
    experimentsSNR = [100, 1];
    
    n = 500; % we generate this data, then subsample it to get 100, 300 data points.
    
    currentLine = 0; % line in results cell we write to (start at line 2). 
    
    fileprefix = 'results/SyntheticDataEvaluation/';

    system([' mkdir -p ', fileprefix]);
    
    fileprefix = [fileprefix, 'synthResults2'];
             
    sampler1 = randsample(1:500, 300);
    sampler2 = randsample(1:300, 100);

    for i = 1:length(covFunction)
        
        % draw data from GP prior
        X = (rand(2*n,dims{i})-0.5)*4;
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
                        
            % create training and testing sets
            X_trn = X(1:n, : );
            y_trn = yn(1:n);
            X_tst = X((n+1):(2*n), :);
            y_tst = yn( (n+1): (2*n) );
            y_tstPreNoiseOriginal = yNoiseLess( (n+1): (2*n) );
            
            disp(['now ', num2str(SNR), ' ', decodeKernelName( encoders{i} )] );
            
            % Calculate kernel and Bayes optimal rates
            bayesOptimalRates = calculateAcc(y_tstPreNoiseOriginal, X_tst, y_tst)   %#ok Same for all three data set sizes
            
            if (SNR == 1)
                bayesValues(i, 1) = bayesOptimalRates;
            elseif (SNR == 100)
                bayesValues(i, 2) = bayesOptimalRates;
            end
            
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
                
                [~,~,~,~,lp] = gp(trueHyperparams, inferenceMethod, [], covFunction{i}, likelihoodFunction, X_trn, y_trn, X_tst, ones(size(y_tst)));
                kernelOptimalRates = calculateAcc(lp, X_tst, y_tst);
                
               % Then do structure discovery:
                
                evalc('[~, ~, ~, ~, ~, ~, ~ , testAccuracy, ~, finalEncoder ] = structureSearch(X_trn, y_trn, X_tst, y_tst, 4 * dims{i}, numRestarts, runParallel, inferenceMethod, likelihoodFunction, searchCriterion);');
                
                currentLine = currentLine + 1;
                
                disp(['Evaluated ', num2str(currentLine - 1), ' out of ', num2str(experimentCount * 3), ' experiments.']);
                
                if sample == 300
                    results{currentLine, 1} = ['$', decodeKernelToLatex(encoders{i}), '$ &' ];
                else 
                    results{currentLine, 1} = '&' ;
                end
                
                if SNR == 1
                    
                    results{currentLine, 2} =  [num2str(sample), '& ' ];
                    results{currentLine, 3} = ['$', decodeKernelToLatex( finalEncoder ), '$ &' ];
                    results{currentLine, 4} = [ num2str(round( testAccuracy * 10000) / 100 ), '\% & '];
                    results{currentLine, 5} = [ num2str(round( kernelOptimalRates * 10000) / 100), '\% & '];
                    
                elseif SNR == 100
                    
                    results{currentLine, 6} =  [num2str(sample), '& ' ];
                    results{currentLine, 7} = ['$', decodeKernelToLatex( finalEncoder ), '$ &' ];
                    results{currentLine, 8} = [ num2str(round( testAccuracy * 10000) / 100 ), '\% & '];
                    results{currentLine, 9} = [ num2str(round( kernelOptimalRates * 10000) / 100), '\% \\ '];
                    
                end
                                           
                fileprefixAux = 'results/SyntheticDataEvaluation/synthResultsTemp2.mat';
                save(fileprefixAux, 'results', 'bayesValues');
           
            end
            
            if SNR == 100
                currentLine = currentLine - 3; % go three lines back to write in 1
            end
            
        end
        
    end

  
    results2  = cell(experimentCount * 3, 9);
    
    for i = 1:3:experimentCount * 3
        for j = 1:9
            results2{i+2, j} = results{i, j};
        end
    end
    
    for i = 3:3:experimentCount * 3
        for j = 1:9
            results2{i-2, j} = results{i, j};
        end
    end
    for i = 2:3:experimentCount * 3
        for j = 1:9
            results2{i, j} = results{i, j};
        end
    end
    
    save( [ fileprefix, '.mat' ], 'results', 'results2', 'bayesValues');
    
end