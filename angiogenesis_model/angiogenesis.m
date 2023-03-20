%% Angiogenesis
 
randn('state',100)
T = 1; N = 500; dt = T/N;
%  first row x, second row y
dW = sqrt(dt) * randn(2, N);

R = 4; j = 2;

Winc =  sum(dW(:,R*(j-1)+1:R*j),2)