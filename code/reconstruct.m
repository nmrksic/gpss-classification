function [w1,w2] = reconstruct(val)

% val = w1 / w2
% w1 + w2 = 1

% val = (1-w2) / w2 = 1 / w2 - 1

w2 = 1 / (val + 1)

w1 = 1 - w2