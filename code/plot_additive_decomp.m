function plot_additive_decomp( X, y, post, kernel_components, componentDims, kernel_params, savefigs, fileprefix, dimensionLabels )
%
% Decomposes an additive GP model into component parts
%
% X,y: The original data
% kernel_components: a cell array of covariance functions, to be added
%                    together.  This code can figure out which dimension
%                    each kernel applies to.
% kernel_params: a cell array of kernel hyperparameters.
%
% David Duvenaud
% Nikola Mrksic
% March 2014


addpath(genpath( 'utils' ));
addpath(genpath( 'gpml' ));

if nargin < 9
    dimensionLabels = cell(size(X, 2), 1); 
    for i = 1 : size(X, 2)
        dimensionLabels = ['Dimension ', num2str(i)]; % if label names are not provided
    end
end

if nargin < 8; fileprefix = ''; end
if nargin < 7; savefigs = false; end





% How much to extend the range of the plots beyond the range of the data.
left_extend = 0.1;
right_extend = 0.1;



[N,D] = size(X);
num_components = length(kernel_components);
assert(num_components == length(kernel_params));



% Next, show the posterior of each component, one at a time.
for i = 1:num_components
    
    
    cur_cov = kernel_components{i};
    cur_hyp = kernel_params{i};    
    
    % Figure out how many and which dimensions this kernel applies to.
    
    currentComp = componentDims{i}; 
    
    cur_ds = currentComp ( 1 : sum ( currentComp > 0 ) );
    num_dims = sum ( currentComp > 0 );
    
    if num_dims >= 3
        fprintf('\n\n Sorry, cant show that high a dimension yet.\n\n');
    end
    

    nstar = 300;
    
    x_left = min(X(:, cur_ds(1))) - (max(X(:,cur_ds(1))) - min(X(:,cur_ds(1))))*left_extend;
    x_right = max(X(:,cur_ds(1))) + (max(X(:,cur_ds(1))) - min(X(:,cur_ds(1))))*right_extend;
    xrange = linspace(x_left, x_right, nstar)';
    xstar = NaN(nstar,D);
    xstar(:, cur_ds(1)) = xrange;
    
    if num_dims == 2
      
        nstar = 50;
        
        x_left = min(X(:, cur_ds(1))) - (max(X(:,cur_ds(1))) - min(X(:,cur_ds(1))))*left_extend;
        x_right = max(X(:,cur_ds(1))) + (max(X(:,cur_ds(1))) - min(X(:,cur_ds(1))))*right_extend;
        xrange = linspace(x_left, x_right, nstar)';
        xstar = NaN(nstar,D);
        
        x2_left = min(X(:,cur_ds(2))) - (max(X(:,cur_ds(2))) - min(X(:,cur_ds(2))))*left_extend;
        x2_right = max(X(:,cur_ds(2))) + (max(X(:,cur_ds(2))) - min(X(:,cur_ds(2))))*right_extend;
        x2range = linspace(x2_left, x2_right, nstar)';
        
        nstar = nstar * nstar;
        xstar = NaN(nstar,D);
        [a,b] = meshgrid(xrange, x2range);
        xstar(:, cur_ds) = [a(:), b(:)];
        
    end


    % Compute Gram matrices of just this component.
    component_sigma = feval(cur_cov{:}, cur_hyp, X);
    component_sigma_star = feval(cur_cov{:}, cur_hyp, xstar, X);
    component_sigma_starstar = feval(cur_cov{:}, cur_hyp, xstar, 'diag');
    
    % Compute posterior of just this component.
    
    component_mean = component_sigma_star * post.alpha;
    
   % component_mean = component_sigma_star * inv(complete_sigma) * y;
   % component_mean = component_sigma_star * (complete_sigma \ y);
    
    
    %component_var = component_sigma_starstar - component_sigma_star / complete_sigma * component_sigma_star';
    
    L = post.L; 
    
  %  L = chol(complete_sigma);  % L will come from approximate inference alg
    %V = L' \ component_sigma_star';
    V = L'\(repmat(post.sW,1,nstar)'.*component_sigma_star)';
    component_var = component_sigma_starstar - diag(V'*V);
    
    data_mean = component_sigma * post.alpha; % / complete_sigma * y;  % complete_sigma * y replaced by alpha from gpml
    
    %data_mean = y - (complete_sigma - component_sigma) / complete_sigma * y;


    % Plot posterior mean and variance.
    %figure(i); clf;
    filename = sprintf( '%sComponent: SE%d', fileprefix, num2str(cur_ds) );
    
    subplot(1, num_components, i);
    
    if num_dims == 1
        nice_oned_plot( X(:,cur_ds), y, data_mean, xrange, component_mean, component_var, savefigs, filename, dimensionLabels, componentDims{i})
    elseif num_dims == 2
        nice_twod_plot( a,b,X(:,cur_ds), y, data_mean, xstar(:, cur_ds), component_mean, component_var, savefigs, filename, dimensionLabels, componentDims{i})
    end
    
        
end

%complete_mean = (complete_sigma - noise_cov) / complete_sigma * y;
%rs = 1 - var(complete_mean - y)/ var(y)



end


function nice_oned_plot( X, y, y_adjusted, xstar, mean, variance, savefigs, filename, dimensionLabels, componentDims)
% Makes a nice plot of a GP posterior.
%
% David Duvenaud
% March 2014

    % How figure looks.
    num_quantiles = 10;

    if nargin < 7; savefigs = false; end
    
    quantiles = linspace(0,0.5,num_quantiles+1);
    quantiles = quantiles(2:end);

    for s = quantiles
        edges = [mean + norminv(s, 0, 1).*sqrt(variance); ...
         flipdim(mean - norminv(s, 0, 1).*sqrt(variance),1)]; 
        h_gp_post = fill([xstar; flipdim(xstar,1)], edges, color_spectrum(2*s), ...
                   'EdgeColor', 'none'); hold on;
    end    
    
    h_data_orig = plot( X(y<0), y(y<0), 'bo', 'Linewidth', 1.5, 'Markersize', 10); hold on;
    h_data_orig = plot( X(y>0), y(y>0), 'b+', 'Linewidth', 1.5, 'Markersize', 10); hold on;

%    h_data_adjust = plot( X, y_adjusted, 'b', 'Linewidth', 1.5, 'Markersize', 10); hold on;
    
    
    %ylim( ylimits);
    xlim( [xstar(1), xstar(end)]);
    %zlim( [xstar(1), xstar(end)]);

    xlabel(dimensionLabels{componentDims(1, 1)});
    ylabel('Class label');
    set(gca,'Layer','top')   % Show the axes again
    %set( gca, 'XTick', [] );
    %set( gca, 'yTick', [] );
    %set( gca, 'XTickLabel', '' );
    %set( gca, 'yTickLabel', '' );
    set(get(gca,'XLabel'),'Interpreter','latex', 'Fontsize', 16);
    set(get(gca,'YLabel'),'Interpreter','latex', 'Fontsize', 16);
    set(gcf, 'color', 'white');

    if savefigs
        saveas(gcf, filename, 'png');
        saveas(gcf, filename, 'fig');
        save2pdf(filename); 
    end
end


function nice_twod_plot( a,b,X, y, y_adjusted, xstar, mean, variance, savefigs, filename, dimensionLabels, componentDims)
% Makes a nice 2D plot of a GP posterior.

    if nargin < 8; savefigs = false; end
        
    %quantiles = linspace(0,0.5,num_quantiles+1);
    %quantiles = quantiles(2:end);

    %for s = quantiles
    %    edges = [mean + norminv(s, 0, 1).*sqrt(variance); ...
    %     flipdim(mean - norminv(s, 0, 1).*sqrt(variance),1)]; 
    %    h_gp_post = fill([xstar; flipdim(xstar,1)], edges, color_spectrum(2*s), ...
    %               'EdgeColor', 'none'); hold on;
    %end    
    
    nstar = sqrt(length(xstar));
    contour( a,b,reshape(mean, nstar, nstar) ); hold on;
    h_data_orig = plot3( X(y<0,1), X(y<0, 2), y(y<0), 'bo', 'Linewidth', 1.5, 'Markersize', 10); hold on;
    h_data_orig = plot3( X(y>0,1), X(y>0, 2), y(y>0), 'r+', 'Linewidth', 1.5, 'Markersize', 10); hold on;
    %h_data_adjust = plot3(  X(:,1), X(:, 2), y_adjusted, 'b+', 'Linewidth', 1.5, 'Markersize', 10); hold on;
    
    %ylim( ylimits);
    %xlim( [xstar(:,1), xstar(end)]);
    
    xlabel(dimensionLabels{componentDims(1, 1)});
    ylabel(dimensionLabels{componentDims(1, 2)});
    zlabel('Class label');

    set(gca,'Layer','top')   % Show the axes again
    %set( gca, 'XTick', [] );
    %set( gca, 'yTick', [] );
    %set( gca, 'XTickLabel', '' );
    %set( gca, 'yTickLabel', '' );
    set(get(gca,'XLabel'),'Interpreter','latex', 'Fontsize', 16);
    set(get(gca,'YLabel'),'Interpreter','latex', 'Fontsize', 16);
    set(get(gca,'ZLabel'),'Interpreter','latex', 'Fontsize', 16);

    set(gcf, 'color', 'white');


    if savefigs
        saveas(gcf, filename, 'png')
        saveas(gcf, filename, 'fig')
        save2pdf(filename); 
    end
end



function col = color_spectrum(p)
    no_col = [1 1 1];
    full_col = [ 1 0 0 ];
    col = (1 - p)*no_col + p*full_col;
end

