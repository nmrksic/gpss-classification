function ret = task1()

meanfunc = {@meanSum, {@meanLinear, @meanConst}};

hyp.mean = [0.5, 1];

ret = 0;