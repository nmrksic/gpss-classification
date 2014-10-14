function [minDist, maxDist] = lengthscales( X )
% calculates the minimal and maximal distance between points in X, for each dimension; non-log.

    n = size(X, 1);
    m = size(X, 2); 

    minDist = zeros(m, 1); 
    maxDist = zeros(m, 1); 


    for dim = 1 : m

        values = sort( unique( X(:, dim) ) );
        
        len = size(values, 1); 
        
        maxDist(dim) = values(len) - values(1);
        values2 = circshift(values, 1);
        differences = values - values2;	
        minDist(dim) = min ( differences(2:len) ); 
        % Smallest distance between non-unique values - definitely the largest completely safe minimal lengthscale we allow.  
        
        
        % We also tried to use the smallest distance spanning any
        % consecutive 10% of the data.
        
        %         values = sort(  X(:, dim) );
        %
        %         span = fix(n/10);
        %
        %         minSpan = 100000000;
        %
        %         for iter = 1 : (n - span)
        %
        %             currSpan = values(iter + span) - values(iter);
        %
        %             if currSpan < minSpan
        %                 minSpan = currSpan;
        %             end
        %
        %         end
        %
        %
        %         if minSpan > minDist(dim)
        %             minDist(dim) = minSpan;
        %         end

        
    end
    
    
end


