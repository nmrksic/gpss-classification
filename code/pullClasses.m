function [XR] = pullClasses(X, threshold)
% this function takes an array od continuous values in [-1,1] and returns
% 1 in each position where the value is >= 0, otherwise -1

% threshold is used if we want to binarize with some biased probability, as
% in when picking out the negativeSet in sEM

if nargin < 2
  threshold = 0;
end

XR = ones(1, length(X));

for i = 1:length(X)
    if(X(i)<=threshold) 
        XR(i) = -1;
    end
end

