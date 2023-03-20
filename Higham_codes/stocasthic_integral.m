% Approximate Stochastic integrals

randn('state',100)

T = 1; N=500; dt = T/N;

dW = sqrt(dt)*randn(1,N);
W = cumsum(dW);

% ito stochastic integral
ito = sum([0,W(1:end-1)].*dW);

