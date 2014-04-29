

idx= 0;
lengthscale = [ 0.1:0.1:50 ];

nlml = zeros(size(lengthscale));

for l = lengthscale
    hyp.mean = 0;
    hyp.cov = log([l 1]);
    idx = idx + 1; 
    temp  = gp(hyp, @infLaplace, @meanConst, kernel, @likErf, xtr, ytr);
    nlml(idx) = temp;
    display(l);
end

plot(lengthscale, nlml);

