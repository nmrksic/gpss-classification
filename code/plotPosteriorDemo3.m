function [] = plotPosteriorDemo3()

n = 500;
D = 3;

InitialiseRand(4);

% Genereate data from known underlying additive function
x = 2 * rand(n, D);
truefunc = @(x) ( sin(3 * x(:,1)) + cos(5 * x(:,2)) + x(:,3) - 1);
sigmoid = @(y) ( 1 ./ ( 1+ exp(-y)));
probs = sigmoid(truefunc(x));
y = probs > rand(n,1);   % Sampling from a Bernoulli for each y.
y = double(y);
y(y==0) = -1;

% plot real functions
    figure(4)
    plot(x(:, 1), sin(3 * x(:,1)), 'b+');
    figure(5)
    plot(x(:, 2), cos(5 * x(:,2)), 'b+');
    figure(6)
    plot(x(:, 3), x(:,3)-1, 'b+');

    
% Set up GP model.
inf = @infLaplace;
mean = @meanConst;
%lik = @likErf;
lik = @likMix;



encoder = [ 1;2;3 ]; 
cov = encodeKernel(encoder, D);

% Set up initial hypers.
Hyp.mean = 0.0;
Hyp.cov = [0 0 0 0 0 0];

% Optimize hypers
hypN = minimize(Hyp, @gp, -100, inf, mean, cov, lik, x, y);

% Unpack covariance function into additive components.
for i = 1:3
    encoder2 = [i];
    covStruct{i} = encodeKernel(encoder2, size(x, 2));
    Hyp2.cov =  hypN.cov( i*2-1 : i*2 )
    hypStruct{i}= Hyp2.cov;
end

% Plot approximate posterior decomposition of latent function.
[~, ~, ~, ~, ~, post] = gp(Hyp, inf, mean, cov, lik, x, y, x, y);
plot_additive_decomp(x, y, post, covStruct, hypStruct, false, true, 'images/classification-decomp');

