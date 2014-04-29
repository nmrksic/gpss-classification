function [bestScores, bestHyper, encoderMatrices] = parallel_bases( X, y, numRestarts, inferenceMethod, likelihoodFunction, searchDepth, searchCriterion ) 
%  This function performs the base kernel evaluation on the cluster. 
%
%
%   Nikola Mrksic
%   April 2013
%
 
    dataSize = size(X, 1);
    dataDim = size(X, 2);

    kernelScores = zeros(numRestarts * dataDim, 1); % the score for a kernel: originally BIC, can be cross-validated accuracy as an alternative. 
    crossValidatedAccuracies = zeros(numRestarts * dataDim, 1);  % cross-validated test accuracies 
    hyperParameters = cell(numRestarts * dataDim, 1);   
    bicValues = zeros(numRestarts * dataDim, 1); % 
    encoderMatrices = cell(dataDim, 1);  
    
    fileID = fopen('randomRestartScript.m', 'r');
    scriptCode = fscanf(fileID, '%s');
    fclose(fileID); 

    kernelCount = dataDim; % number of structurally different kernels to expand
  
    [minDist, maxDist] = lengthscales(X);
    
    for i = 1:dataDim 

        encoderMatrix = zeros(10, 10); 
        encoderMatrix(1,1) = i;
        encoderMatrices{i} = encoderMatrix;

    end
 
    finalHyperParams = cell(numRestarts * kernelCount, 1); 
 
    for i = 1 : (kernelCount)
        for j = 1 : numRestarts
            
            idx = (i-1) * numRestarts + j;
            
            finalHyperParams{idx}.mean = 0;
            finalHyperParams{idx}.cov = log( [minDist(i) + ( maxDist(i) - minDist(i) ) * rand(1, 1), 5 * rand(1,1) + 1] );
           
            % if it's a likelihood mixture (currently the only non - likErf
            % likelihood supported, initialise its hyperparameters:
            if  iscell(likelihoodFunction) == 1
                finalHyperParams{idx}.lik = [ -1 + randn(), 1 + randn() ];
            end
                       
        end
    end
  
    disp( ['Evaluating the base SE kernels, number of experiments is: ', num2str(numRestarts * kernelCount), '.' ]); 
  
    system('mkdir -p scripts'); % ensure that the scripts folder is available. 
    
    for i = 1:(kernelCount)
        for j = 1:numRestarts
            
            idx = (i-1) * numRestarts + j;
            
            if ( iscell(likelihoodFunction)==0 ) % if not a cell, then it's likErf, so no hyperparameters. 
                
                script_code = ['i=', num2str(idx), ';likelihoodFunction=@likErf;', 'covFunction=', decodeKernelCode(encoderMatrices{i}, size(X,2)), ';hyperParameters.mean=0;hyperParameters.cov=[', num2str(finalHyperParams{idx}.cov), '];',  scriptCode];
                
            else
                
                script_code = ['i=', num2str(idx), ';likelihoodFunction={@likMix,{@likUni,@likErf}};', 'covFunction=', decodeKernelCode(encoderMatrices{i}, size(X,2)), ';hyperParameters.mean=0;hyperParameters.cov=[', num2str(finalHyperParams{idx}.cov), '];hyperParameters.lik=[', num2str(finalHyperParams{idx}.lik) ,'];', scriptCode];
                
            end
            
            fileID = fopen(['scripts/script', sprintf('%09d', idx), '.m'], 'w');
            fprintf(fileID, '%s\n', script_code);
            fclose(fileID);
                
        end
    end
 
    save('data/data.mat', 'X', 'y');

    [~,~] = system('python runscriptsinparallel.py'); % supress output from the call to the cluster
 
 % Load output data:
 
     for i = 1:(kernelCount)
         for j = 1 : numRestarts

             idx = (i-1) * numRestarts + j;

             file_name = ['outputs/script',sprintf('%09d', idx),'.mat'];
             load(file_name);
                
             % Load data received:
             bicValues( idx ) = BIC(bicValue, encoderMatrices{i}, dataSize); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)
             hyperParameters{ idx } = hypN;
             % Compute cross validated accuracy:
             covFunction = encodeKernel( encoderMatrices{i} ,  dataDim );
             crossValidatedAccuracies(idx) = crossValidatedAccuracy(X, y, covFunction, hypN, inferenceMethod, likelihoodFunction);
             
             
            
             % --------- Prune small lengthscales -----------------------
             newHyperParamAll = hypN.cov;
             newHyperParam = newHyperParamAll(1:2:end); % take only lengthscales
             
             encoderDims = encoderMatrices{i};
             encoderDims = encoderDims';
             
             encoderDims = encoderDims(encoderDims > 0);
             
             for i2 = 1:size(encoderDims) % if any of the dimensions went under, stop considering this kernel
                
                 if exp( newHyperParam (i2) ) < minDist ( encoderDims ( i2 ) ) 
                     bicValues(idx) = 2000000000;
                     crossValidatedAccuracies(idx) = 0;
                 end
             end
             % ----------------------------------------------------------
             
             
             
        end

     end

    [~,~] = system('rm outputs/*');
    
    if searchCriterion == 0
        
        kernelScores = bicValues;
    
    elseif searchCriterion == 1
        
        kernelScores = crossValidatedAccuracies;
        
    end
    
   bestScores = zeros(1, searchDepth);
   bestHyper = cell(1, searchDepth);

   % Choose the best hyperparameters for each of the base kernels:
   for i = 1 : kernelCount
        
        [bestScores(i), idx] = min( kernelScores ( (i-1) * numRestarts + 1 : i * numRestarts ) );
        bestHyper{i} = hyperParameters{(i-1)* numRestarts + idx};
        
   end
    
   
end
    
    