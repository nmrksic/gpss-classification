
fileID = fopen('randomRestartScript.m', 'r');
scriptCode = fscanf(fileID, '%s');
fclose(fileID); 

kernelCount = dim;

[minDist, maxDist] = lengthscales(X);

 for i = 1:dim 

     encoderMatrices(i, 1, 1) = i;

     encoderMatrix(1, 1) = i;
     
     covFunc{i} = encodeKernel(encoderMatrices(i, :, :), dim);
     
     covFunctions{i} = encodeKernel(encoderMatrices(i, :, :), dim);
    
 end
 
  finalHyperParams = cell(1,  numExp * kernelCount ); 
  covFunctions = cell (1,  numExp * kernelCount ); 

 
   for i = 1 : (kernelCount)
       for j = 1 : numExp
      
           idx = (i-1) * numExp + j;
           
           finalHyperParams{idx}.mean = 0;
           finalHyperParams{idx}.cov = log( [minDist + (maxDist - minDist) * rand(1, 1), 10 * rand(1,1)] ); 
           
           covFunctions{idx} = covFunc{i}; 
           
       end
   end
  
   disp( ['Evaluating the base kernels, # experiments = ', num2str(numExp * kernelCount)] ); 
   
   encodersBackup = encoderMatrices;
   
   encoderMatrices = cell(1, 1000); 
   
   for i = 1 : kernelCount*numExp
       encoderMatrices{i} = squeeze(encodersBackup(i, :, :));
   end
   
   parallel_call
   
   encoderMatrices = encodersBackup;
   
   % by now, this should get rid of all the scripts it created. 
   
   for i = 1 : kernelCount
        
        [bestBicVals(i), indeks] = min( bicValues ( (i-1) * numExp + 1 : i * numExp ) );
        
        % disp(['current', num2str(i), num2str(indeks), 'range: ', num2str((i-1) * numExp + 1 : i * numExp )]); 
        
        bestHyper{i} = hyperParameters{(i-1)* numExp + indeks};
        
   end
    
   
   % now to adapt to rest of the code in Automated Statistician:
   
   covFunctions =  cell(1, searchSteps);
   
    for i = 1:dim % initialise the base kernels and determine their BICs and test accuracies

     covFunctions{i} = encodeKernel(squeeze(encoderMatrices(i, :, :)), dim);
     
     
     kernelNames{i} = ['SE', num2str(i)];

     [~,~,~,~,lp] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunctions{i}, likfunc, X, y, X_tst, ones(size(y_tst)));
     predictiveAccuraccies(i) = calculateAcc(lp, X_tst, y_tst);
     
     % train accs, alternative search criterion:
     [~,~,~,~,lp2] = gp(bestHyper{i}, inferenceMethod, meanfunc, covFunctions{i}, likfunc, X, y, X, ones(size(y)));
     trainAccuracies(i) = calculateAcc(lp2, X, y);
     % ....
            
    end
 

    
    