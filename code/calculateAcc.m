function [accuracy] = calculateAcc(lp, X, y)

 predictions = pullClasses(exp(lp), 0.5); % turn probabilities into +-1 predictions

 errors = sum ( abs(predictions' - y) ) / 2; % number of prediction errors made
 
 accuracy = 1 - errors / size(X, 1);

end