%% Test strong convergence of Euler-Maruyama
% the SDE is dX = lambda * X * dt + mu*X*dW, X(0)=Xzero
% where lambda = 2, mu = 1, Xzero = 1
% discretized brownian path over [0,1] has dt = 2^(-9)
% E-M uses 5 different timesteps: 16dt, 8dt, 4dt, 2dt, dt.
% Examnine strong convergence at T=1: E|X_L - X(T)|

randn('state',100)
% problem parameters
lambda = 2; mu = 1; Xzero = 1;
T = 1; N = 2^9; dt = T/N;

% number of paths

M = 1000;

% prealocation of the array
Xerr= zeros(M,5);
%iteration over the M brownian paths
for s = 1:M
    dW = sqrt(dt)*randn(1,N); %brownian increment
    W = cumsum(dW); %sample over discrete brownian path
    % solution
    Xtrue = Xzero * exp((lambda - 0.5*mu^2)+mu*W(end));

    % generation of numerical solution over 5 different timesteps
    for p = 1:5
        R = 2^(p-1); Dt = R * dt; L = N/R; % L euler steps of size Dt
        Xtemp = Xzero;
        % generation of the numerical array of solution
        for j = 1:L
            Winc = sum(dW(R*(j-1)+1:R*j));
            Xtemp = Xtemp + Dt*lambda * Xtemp + mu*Xtemp*Winc;

        end
        Xerr(s,p)=abs(Xtemp - Xtrue); % stpre the error at t = 1
    end
end

Dtvals = dt*(2.^([0:4]) );
subplot(221)
loglog(Dtvals,mean(Xerr),'b*-'), hold on
loglog(Dtvals,(Dtvals.^(.5)),'r--'),hold off % reference slope of 1/2

axis([1e-3 1e-1 1e-4 1])
xlabel('\Delta t'), ylabel('Sample average of |X(T)-X_L|')
