clc, close all
addpath("./functions/")
randn('state',100)


% Initialize time steps
T = 1; % time from 0 to 1
N = 500; % there are going to be 500 steps
dt = T/N; % the discrete steps of the brownian
%brownina motion
x_100 = []
for i = 1:100
    dW = sqrt(dt)*randn(1,N);
    
    W = cumsum(dW);
    
    % brownian motion parameters
    R = 4;
    Dt = R*dt;
    L = N/R;
    
    % function parameters
    a = 10;
    
    % initiation
    
    Xtemp = 0;
    Xem = [];
    Xzero = 0;
    
    for j = 1:L
        winc = sum(dW(:,R*(j-1)+1:R*j));
        Xtemp = Xtemp + a * Dt + winc;
        Xem(j) = Xtemp;
    end 
    x_100(i,:) = Xem;
end

x_100_mean = mean(x_100)

%plot([0:1:L],[0,x_100_mean])
plot([0:1:L],[0,x_100(1,:)])
hold on
for i = 2:length(x_100)
    plot([0:1:L],[0,x_100(i,:)])
end