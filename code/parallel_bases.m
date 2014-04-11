function [bestBicVals, bestHyper, encoderMatrices] = parallel_bases(X, y, numExp, inferenceMethod, searchSteps)

encoderMatrices = zeros(searchSteps, 10, 10); 
dim = size(X, 2);

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
      
           idx2 = (i-1) * numExp + j;
           
           finalHyperParams{idx2}.mean = 0;
           finalHyperParams{idx2}.cov = log( [minDist + (maxDist - minDist) * rand(1, 1), 10 * rand(1,1)] ); 
           
           covFunctions{idx2} = covFunc{i}; 
           
       end
   end
  
   disp( ['Evaluating the base SE kernels, number of experiments is: ', num2str(numExp * kernelCount),'.'] ); 
   disp(' ');
   
   encodersBackup = encoderMatrices;
   
   encoderMatrices = cell(1, 1000); 
   
   for i = 1 : kernelCount
       encoderMatrices{i} = squeeze(encodersBackup(i, :, :));
   end
%   ------------------------------------------------------------------
  system('mkdir -p scripts'); % make sure we have the scripts folder available. 

for i2 = 1:(kernelCount)
    for j3 = 1:numExp
        
        i3 = (i2-1)*numExp + j3;
    
     script_code = ['i=', num2str(i3), ';', 'covFunction=', decodeKernelCode(encoderMatrices{i2}, size(X,2)), ';hyperParameters.mean=0;hyperParameters.cov=[', num2str(finalHyperParams{i3}.cov), '];'  scriptCode];
     %save(['scripts/script', sprintf('%09d', i3), '.m'], 'script_code');
     
     fileID = fopen(['scripts/script', sprintf('%09d', i3), '.m'], 'w');
     fprintf(fileID, '%s\n', script_code);
     fclose(fileID);
    end
 end
 
 save('data/data.mat', 'X', 'y');

  [A,B] =  system('python runscriptsinparallel.py');
 
 % Load data
 
  for i2 = 1:(kernelCount)
       for j1 = 1 : numExp

        file_name = ['outputs/script',sprintf('%09d', (i2-1)*numExp + j1),'.mat'];

        load(file_name);
        
        bicValues( (i2-1) * numExp + j1) = BIC(bicValue, encoderMatrices{i2}, size(X, 1)); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)
        
        hyperParameters{(i2-1)*numExp + j1} = hypN;

       covFunction = encodeKernel( squeeze(encoderMatrices{i2} ),  size(X, 2) );

     % train accs, alternative search criterion:
     [~,~,~,~,lp2] = gp(hyperParameters{(i2-1)*numExp + j1}, inferenceMethod, @meanConst, covFunction, @likErf, X, y, X, ones(size(y)));
     trainAccs((i2-1)*numExp + j1) = calculateAcc(lp2, X, y); 
        
     %bicValues = trainAccs; % CHANGE FOR USING TRAINING ACCURACY FOR BUILDING IT. 
        

         % disp([num2str(i), ' out of ', num2str(numExp*kernelCount), ' experiments for the current expansion stage. ']);
         % [bicValues(i), hyperParameters{i}] = randomRestarts2( covFunctions{i} , finalHyperParams{i}, X, y, inferenceMethod); % how do we pass data to server? 

       end
  
  end

system(['rm outputs/*']);
 %  ------------------------------------------------------------------
   encoderMatrices = encodersBackup;
   
   % by now, this should get rid of all the scripts it created. 
   
   bestBicVals = zeros(1, searchSteps);
   bestHyper = cell(1, searchSteps);

   for i = 1 : kernelCount
        
        [bestBicVals(i), indeks] = min( bicValues ( (i-1) * numExp + 1 : i * numExp ) );
        
        % disp(['current', num2str(i), num2str(indeks), 'range: ', num2str((i-1) * numExp + 1 : i * numExp )]); 
        
        bestHyper{i} = hyperParameters{(i-1)* numExp + indeks};
        
   end
    
   
   % now to adapt to rest of the code in Automated Statistician:
   
  
 
end
    
    