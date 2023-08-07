clc, close all
addpath("./functions/")
randn('state',1000)

H = 0.1;   % .5 = Brownian motion; 0 = negative correlated; 1 = positive correlated; 1 > H > .5 = inbewteen
nReps = 1e1;
Xdata{nReps} = [];
Da_data{nReps} = [];


% Initialize time steps
T = 3; % time from 0 to 1
N = 2000; % there are going to be 500 steps
dt = T/N; % the discrete steps of the brownian



for ix = 1:nReps    
    % based on the empirical data extracted
    % from stokes and lauffenburger
    beta = 1/3; % h^-1
    alpha = 40 * 1/3; %µm^2 h-3
    % we define this parameter arbitrarialy
    kappa = 10 ;
    
    %Fractional Brownian Path
    %Now we define a fractional browian motion
    dW = sqrt(dt) * randn(2,N);
    W = cumsum(dW);
    
    %Now we plot the brownian walk for the x and y variables to see if they make sense
    % plot([0:dt:T],dW(1,:)), hold on
    % plot([0:dt:T],dW(2,:)), hold off
    
    %Begining of the modeling
    %To solve the stochastic differential equation model we define the following L interger and stepsize
    R = 4;
    Dt = R*dt;
    L = N/R;
    % We initialize the solution vector
    Xem = zeros(2,L); % first x
    % second y
    Vem = zeros(2,L);
    
    % Initialize the first point
    Xzero = [0;0];
    Xtemp = Xzero;
    Vzero = [0;0];
    Vtemp = Vzero;
    % And we initialize the first angles
    theta = pi/2;
    phi = pi/2;
   
    
   % These equations have been implemented in the functions that are in the folder functions

    % plot([1:100],da(:))    
    % We beging the for loop for the model
    Theta = [];
    DA_list = [];
    for j = 1:L
        winc = sum(dW(:,R*(j-1)+1:R*j),2);
        %winc = dW(j);
        DA = gradient_div(Xtemp(1,:),Xtemp(2,:),2e-14);
        Vtemp = Vtemp + (-beta * Vtemp + kappa * DA + sin(abs(phi/2)))*Dt + sqrt(alpha)*winc;
        Vem(:,j) = Vtemp;
        Xtemp = Xtemp + Vtemp * Dt;
        Xem(:,j) = Xtemp;
        if j ~= 1
            xj = Xem(1,j);
            yj = Xem(2,j);
            xj_1 = Xem(1,j-1);
            yj_1 = Xem(2,j-1);
            theta = atan((yj-yj_1)/(xj-xj_1));
            phi = calculate_phi( xj, yj , xj, 3, theta);
            Theta = [Theta,theta];
        end
        DA_list(j) = DA;
    end
    % We see the array of solutions for the position    
    tSpan = linspace(0,1,L);    
    Xdata{ix} = [Xem(1,:);Xem(2,:);tSpan];    
    DA_data{ix} = [DA_list; tSpan];

end
%plot(Theta)
%return
figure(4)
cm = winter(nReps);
for jx = 1:nReps
    
    name = ['Reps =',num2str(jx)];
    da_rep = DA_data{jx};
    plot(da_rep(2,:),da_rep(1,:),'Color',cm(jx,:),'LineStyle','-.','DisplayName',name);
    hold on
end
axis tight
xlabel("Time")
legend show
ylabel("Gradient Value")

figure(1)
for jx = 1:nReps
    Xtemp = Xdata{jx};
    name = ['Reps =',num2str(jx)];
    plot(Xtemp(1,:),Xtemp(2,:),'Color', cm(jx,:),'LineStyle','-.','DisplayName',name, ...
        'LineWidth',2);
    hold on
end
axis tight
legend show
xlabel('X position')
ylabel('Y position')
title('Angiogenesis based on Brownian Motion')

figure(2)
for jx = 1:nReps
    Xtemp = Xdata{jx};
    name = ['Reps =',num2str(jx)];
    plot(Xtemp(3,:),Xtemp(1,:),'Color',cm(jx,:),'LineStyle','-.','DisplayName',name);
    hold on
end
axis tight
xlabel('time')
legend show
ylabel('X position')

figure(3)
for jx = 1:nReps
    Xtemp = Xdata{jx};
    name = ['Reps =',num2str(jx)];
    plot(Xtemp(3,:),Xtemp(2,:),'Color',cm(jx,:),'LineStyle','-.','DisplayName',name, ...
        'LineWidth', 2);
    hold on
end
axis tight
legend show
xlabel('time')
ylabel('Y position')

