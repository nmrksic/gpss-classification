%% Nikola's code - using meanConst

hyp.mean = -11.2476;
hyp.cov = [6.1526 0.7555];
covFunSE1 =   {@covMask, {[1 0 0], {@covSEiso}}};
nlmlLaplace = gp(hyp, @infLaplace , @meanConst, covFunSE1, @likErf, X, y');
nlmlEP = gp(hyp, @infEP , @meanConst, covFunSE1, @likErf, X, y');
nlmlVB = gp(hyp, @infVB , @meanConst, covFunSE1, @likErf, X, y');

display([nlmlVB, nlmlEP, nlmlLaplace]);

%% JRL's code - using covConst

hyp.mean = [];
hyp.cov = [6.1526 0.7555, log(11.2476)];
cov_fn =  {@covSum, {{@covMask, {[1 0 0], {@covSEiso}}}, {@covConst}}};
nlmlLaplace = gp(hyp, @infLaplace , @meanZero, cov_fn, @likErf, X, y');
nlmlEP = gp(hyp, @infEP , @meanZero, cov_fn, @likErf, X, y');
nlmlVB = gp(hyp, @infVB , @meanZero, cov_fn, @likErf, X, y');

display([nlmlVB, nlmlEP, nlmlLaplace]);

%% Is there a bug when using VB with a non zero mean function?