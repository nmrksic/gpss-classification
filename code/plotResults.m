function [] = plotResults(expName, numFoldsToPlot, X, y)
% numFolds specifies how many folds to complete - if not all runs are
% complete, this allows us to view the ones completed so far. 
  
    if nargin < 3
        name = ['../Data/classification/', expName, '.mat'];
        load(name);
    end

    dataDim = size(X, 2); % dimensionality of the data

    seed = 0;

    InitialiseRand(seed); % random seed initialised here. 
    
    initialiseDataDimensionLabels; % get dimension labels for breast, pima, liver, heart.  

    [trnX, trnY, tstX, tstY] = crossValidate(X, y, 10); % we are doing 10-fold cross validation

    fileprefix = ['results/', expName, '/', ];     
        
    system([' mkdir -p ', fileprefix]);

    averageAccumulator = 0;
    
    msg = [];
    
    for i = 1:numFoldsToPlot
           
        currentFold = i; % declared just for the sake of saving the data
        
        filePrefixNew = [fileprefix, 'fold', num2str(currentFold) , '/'];
 
        load ( [ filePrefixNew, 'searchStats.mat']);          
         
        numIterations = sum(nnz(BicValsList)); % the number of search stages 

        averageAccumulator = averageAccumulator + testAccuracciesList(numIterations);
                 
        currentAverage = 100 * averageAccumulator / i;
        
        disp(['Currently plotting fold ', num2str(i), ' of ', num2str(numFoldsToPlot), '. Number of stages to be plotted is:', num2str(numIterations-dataDim)]);

        for plott = dataDim + 1 : numIterations
    
            filePrefixPlot = [filePrefixNew, 'Stage', num2str(plott), '/'];
            % close all
            plotPosteriors(trnX{i}, trnY{i}, squeeze( allEncoderMatrices(plott, :, :) ) , hyperList{plott}, filePrefixPlot, dimensionLabels);
            tightfig;

            pause
            
        end
        
        msg = [msg, ' ', num2str(testAccuracciesList(numIterations))];
        
    end

    disp(msg)
    
    disp(' ');
    disp(['The current classification accuracy of ', expName, ' on ', num2str(numFoldsToPlot), ' folds evaluated is ', num2str(currentAverage), '%.']);
    
end
    