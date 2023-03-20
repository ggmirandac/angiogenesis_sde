%% Test Strong convergence of Milstein method
%
% Solves dX = r*X*(K-X) dt + beta*X dW, X(0)=Xzero
% where r = 2; k = 1; beta = 1 and Xzero = 0.5
% Discretized brownian path over [0,1] has dt = 2^(-11)
% Milstein uses timesteps 128*dt, 64*dt, 32*dt, 16*dt (also dt for
% reference)
% Examine strong convergence at T=1; E|X_L-X(T)|

rand('state',100)

% problem parameters
r=2 ; K=1; beta=0.25; Xzero=0.5;
T=1; N=2^(11); dt = T/N;
M = 500; % amount of brownian paths

R = [1; 16; 32 ; 64; 128]; % Misltein stepsize are R* dt

dW = sqrt(dt)*randn(M,N); % M paths and N increments
Xmil = zeros(M,5); % M paths and 5 iterations

for p = 1:5
    Dt = R(p)*dt; L = N/R(p);
    %initiation of M Xzeros 
    Xtemp = Xzero*ones(M,1);
    for j = 1:L
        Winc = sum(dW(:,R(p)*(j-1)+1:R(p)*j),2);
        Xtemp= Xtemp+Dt*r*Xtemp.*(K-Xtemp) + beta*Xtemp.*Winc+0.5*beta^2*Xtemp.*(Winc.^2-Dt);
    end
    Xmil(:,p) = Xtemp; % store milstein solution at t=1
end

Xref = Xmil(:,1);
Xerr = abs(Xmil(:,2:5)-repmat(Xref,1,4));
mean(Xerr);
Dtvals = dt*R(2:5);





