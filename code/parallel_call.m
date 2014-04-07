
%disp( [' currently evaluating the base kernels, # experiments = ', num2str(numExp * kernelCount)] ); 

for i = 1:(numExp*kernelCount)
    
     script_code = ['i=', num2str(i), '; ', scriptCode];
     save(['script', sprintf('%09d', i), '.m'], 'script_code');
     
     fileID = fopen(['scripts/script', sprintf('%09d', i), '.m'], 'w');
     fprintf(fileID, '%s\n', script_code);
     fclose(fileID);
    
     % disp([num2str(i), ' out of ', num2str(numExp*kernelCount), ' experiments for the current expansion stage. ']);
     % [bicValues(i), hyperParameters{i}] = randomRestarts2( covFunctions{i} , finalHyperParams{i}, X, y, inferenceMethod); % how do we pass data to server? 
 end
 
 save('data/data.mat', 'X', 'y', 'covFunctions', 'finalHyperParams');

 meanfunc = @meanConst;
 likfunc = @likErf;
 inferenceMethod = @infLaplace;
 
 %system('python run-scripts-in-parallel.py'); 
 
 system('python runscriptsinparallel.py');
 % Load data
 
  for i = 1:(kernelCount)
       for j = 1 : numExp

        file_name = ['outputs/script' sprintf('%09d', (i-1)*numExp + j) '.mat'];

        load(file_name);
        bicValues( (i-1)*numExp + j) = BIC(bicValue, encoderMatrices{i}, size(X, 1)); % we only calculate BICs here, as the minimizer returns NLMLs (saved as bicValue)
        
        hyperParameters{(i-1)*numExp + j} = hypN;

       covFunction = encodeKernel( squeeze(encoderMatrices{i} ),  size(X, 2) );

 
     % train accs, alternative search criterion:
     [~,~,~,~,lp2] = gp(hyperParameters{(i-1)*numExp + j}, inferenceMethod, meanfunc, covFunction, likfunc, X, y, X, ones(size(y)));
     trainAccs((i-1)*numExp + j) = calculateAcc(lp2, X, y); 
        
     %bicValues = trainAccs; % CHANGE FOR USING TRAINING ACCURACY FOR BUILDING IT. 
        
        system(['rm ' file_name]);

         % disp([num2str(i), ' out of ', num2str(numExp*kernelCount), ' experiments for the current expansion stage. ']);
         % [bicValues(i), hyperParameters{i}] = randomRestarts2( covFunctions{i} , finalHyperParams{i}, X, y, inferenceMethod); % how do we pass data to server? 

       end
  
  end