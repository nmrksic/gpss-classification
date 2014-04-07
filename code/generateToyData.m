function [X, y] = generateToyData ()

n = 300;

X = rand(n, 3);

truefunc = @(x) ( sin(x(:,1)) + cos(x(:,2)) + x(:,3));
sigmoid = @(y) ( 1 ./ ( 1+ exp(-y)));

probs = sigmoid(truefunc(X));

y = probs < rand(n,1);   % Sampling from a Bernoulli for each y.


save( 'structured_classification.mat', 'X', 'y' );