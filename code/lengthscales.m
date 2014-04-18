function [minDist, maxDist] = lengthscales( X )
% calculates the minimal and maximal distance between points in X, for each dimension; non-log.

    n = size(X, 1);
    m = size(X, 2); 

    minDist = zeros(m, 1); 
    maxDist = zeros(m, 1); 


    for dim = 1 : m

        values = sort( X(:, dim) );
        maxDist(dim) = values(n) - values(1);
        values2 = circshift(values, 1);
        differences = values - values2;	
        minDist(dim) = min ( differences(2:n) ); 

    end

