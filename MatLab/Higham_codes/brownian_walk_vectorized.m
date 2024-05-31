% Brownian path simulation: vectorized

randn('state',100) % the state of the random values

T = 1; N = 500; dt=T/N;

% generates a 1,N array of random variables 

dW = sqrt(dt)*randn(1, N);

% Generate the cumulative sum, until the j variable 
% cumulative sum generates the sum to see how it grows over time

W = cumsum(dW);

% plot

plot([0:dt:T],[0,W],'r-');
xlabel('t','FontSize',16)
ylabel('W(t)','FontSize',16,'Rotation',0)

