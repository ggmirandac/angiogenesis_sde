%% Euler-Maruyama method on linear SDE
% the SDE is dX = lambda * X * dt + mu*X*dW, X(0)=Xzero
% where lambda = 2, mu = 1, Xzero = 1
% discretized brownian path over [0,1] has dt = 2^(-8)
% EM uses steps R*dt

randn('state',100)
% problem parameters
lambda = 2; mu = 1; Xzero = 1;
T = 1; N = 2^8; dt = T/N;
% brownian increment
dW = sqrt(dt)*randn(1,N);
% discretized brownian path
W = cumsum(dW);

% exact solution for the SDE, vectorial form

Xtrue = Xzero*exp((lambda - 0.5*mu^2)*([dt:dt:T])+ mu*W);

plot([0:dt:T],[Xzero,Xtrue],'m-'), hold on

% let's generate the numerical solution to this problem

R = 4; Dt = R*dt; L = N/R; % L ME steps of size DT

% prelocate the array for efficiency
Xem = zeros(1,L);

% initial X in time (Xtemp) is Xzero
% I am generating them from time = 1 to time = L
Xtemp  = Xzero ;

for j = 1:L
    Winc = sum(dW(R*(j-1)+1:R*j));
    Xtemp = Xtemp + lambda * Xtemp * Dt + mu * Xtemp * Winc;
    Xem(j) = Xtemp;
end

plot([0:Dt:T],[Xzero,Xem],'r--*'), hold off
xlabel('t','FontSize',12)
ylabel('X','FontSize',16,'Rotation',0,'HorizontalAlignment','right')

emerr = abs(Xem(end)-Xtrue(end));







