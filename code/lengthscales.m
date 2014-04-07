function [minDist, maxDist] = lengthscales( X )
% calculates the minimal and maximal distance between points in X; non-log.

n = size(X, 1);

minDist = norm( X(1, :) - X(2, :) );
maxDist = 0;

for i = 1:n
    for j = 1:n
        if i~=j
      
            distance = norm(  X(i, :) - X(j, :) );
            
            if distance > maxDist
                maxDist = distance;
            end
            
            if distance < minDist
                minDist = distance;
            end
        end
    end
end
