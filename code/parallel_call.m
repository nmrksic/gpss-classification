system('mkdir -p scripts'); % make sure we have the scripts folder available. 

for i2 = 1:(kernelCount)
    for j3 = 1:numExp
        
        i3 = (i2-1)*numExp + j3;
    
     script_code = ['i=', num2str(i3), ';', 'covFunction=', decodeKernelCode(encoderMatrices{i2}, size(X,2)), ';hyperParameters.mean=0;hyperParameters.cov=[', num2str(finalHyperParams{i3}.cov), '];'  scriptCode];
    % save(['scripts/script', sprintf('%09d', i3), '.m'], 'script_code');
     
     fileID = fopen(['scripts/script', sprintf('%09d', i3), '.m'], 'w');
     fprintf(fileID, '%s\n', script_code);
     fclose(fileID);
    end
 end
 
 save('data/data.mat', 'X', 'y');

 [A, B] = system('python runscriptsinparallel.py');
 
 % Load data
 
  for i2 = 1:(kernelCount)
       for j1 = 1 : numExp

        file_name = ['outputs/script' sprintf('%09d', (i2-1)*numExp + j1) '.mat'];

        load(file_name);
        
        bicValues( (i2-1) * numExp + j1) = BIC(bicValue, encoderMatrices{i2}, size(X, 1)); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)
        
        hyperParameters{(i2-1)*numExp + j1} = hypN;

       covFunction = encodeKernel( squeeze(encoderMatrices{i2} ),  size(X, 2) );

     % train accs, alternative search criterion:
     [~,~,~,~,lp2] = gp(hyperParameters{(i2-1)*numExp + j1}, inferenceMethod, meanfunc, covFunction, likfunc, X, y, X, ones(size(y)));
     trainAccs((i2-1)*numExp + j1) = calculateAcc(lp2, X, y); 
        
     %bicValues = trainAccs; % CHANGE FOR USING TRAINING ACCURACY FOR BUILDING IT. 
        

         % disp([num2str(i), ' out of ', num2str(numExp*kernelCount), ' experiments for the current expansion stage. ']);
         % [bicValues(i), hyperParameters{i}] = randomRestarts2( covFunctions{i} , finalHyperParams{i}, X, y, inferenceMethod); % how do we pass data to server? 

       end
  
  end

system(['rm outputs/*']);

  
  % removed from randomRestartScript.m:
  
% 
 
 %covFunction = covFunctions{i};
 %hyperParameters = finalHyperParams{i}; 
