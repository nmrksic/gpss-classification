function [] = plotPosteriors(x, y, encoderMatrix, Hyp, fileprefix, dimensionLabels)
% This function takes a kernel consisiting of a sum of different kernels,
% and then plots each of its 1D components posteriors. 
% As input, the function takes the encoderMatrix and the hyperparameters of
% the final hyperparameter. 

    if nargin < 6
        dimensionLabels = cell(size(x, 2), 1); 
        for i = 1 : size(x, 2)
            dimensionLabels{i} = ['Dimension ', num2str(i)]; % if label names are not provided
        end
    end

    if nargin < 5 
        fileprefix = 'classification_decomposition';
    end

    n = size(encoderMatrix, 1); % number of product terms
    D = size(x, 2); % dimensionality of the data

 
    % Set up GP model.
    inf = @infLaplace;
    mean = @meanConst;
    lik = @likErf;
    %lik = @likMix;

    covFunction = encodeKernel(encoderMatrix, D);


    lengths = zeros(n, 1); % will contain the number of terms in each product term: if 1, we can plot it. 
   
     for i = 1:n

         lengths(i) = nnz(encoderMatrix(i, :));

         if(lengths(i)==0) % n should contain the actual number of product terms
             n = i-1;
             break;
         end

     end

    plottedCount = 0; % how many components we will be plotting.
    disp(' ');

    % Unpack covariance function into additive components.
    for i = 1:size(encoderMatrix, 1)
        
        if lengths(i) == 1 % only if it's a single component do we plot it

            plottedCount = plottedCount + 1;
            
            encoderNew = zeros(1, D);
            
            encoderNew(1,1) = encoderMatrix(i, 1);
            
            covStruct{plottedCount} = encodeKernel(encoderNew, D);
            componentDims{plottedCount} = encoderNew;
            
            previousHyperCount = nnz ( encoderMatrix (1 : (i - 1), : ) ) * 2; 
            Hyp2.cov =  Hyp.cov( previousHyperCount + 1 : previousHyperCount + 2 );
            hypStruct{plottedCount}= Hyp2.cov;
            
            disp(['Plotting the SE', num2str(encoderMatrix(i, 1)), ' kernel posterior']);
            
        elseif lengths(i) == 2
            
            plottedCount = plottedCount + 1;
            
            encoderNew = zeros(1, D);
            
            encoderNew(1,1) = encoderMatrix(i, 1);
            encoderNew(1,2) = encoderMatrix(i, 2);
            
            covStruct{plottedCount} = encodeKernel(encoderNew, D);
            componentDims{plottedCount} = encoderNew;

            previousHyperCount = nnz ( encoderMatrix (1 : (i - 1), : ) ) * 2; 
            Hyp2.cov =  Hyp.cov( previousHyperCount + 1 : previousHyperCount + 4 );
            hypStruct{plottedCount}= Hyp2.cov;
            
            disp(['Plotting the SE', num2str(encoderMatrix(i, 1)), ' X SE', num2str(encoderMatrix(i, 2)) , ' kernel posterior']);

        end
    end

    if (plottedCount > 0)
    
        % Plot approximate posterior decomposition of latent function.
        [~, ~, ~, ~, ~, post] = gp(Hyp, inf, mean, covFunction, lik, x, y, x, y);
        plot_additive_decomp(x, y, post, covStruct, componentDims, hypStruct, true, fileprefix, dimensionLabels);
        
    else
        
        disp (' No 1D components to plot.' );
        
    end
    
end