clc, close all
clear
addpath("./functions/")
randn('seed', 1000)

nReps = 10000;
Xdata{nReps} = [];
Da_data{nReps} = [];
Theta_data{nReps} = [];
Phi_data{nReps} = [];
V_data{nReps} = [];
Time_possition = zeros(1,nReps);
file_name_hist = fullfile('./figures','histogram_brownianmotion.pdf');
file_name_figure = fullfile('./figures','plot_brownianmotion.svg');
file_name_data = fullfile('./data','time_pos_bm.mat');
% Initialize time steps
T = 100; % time from 0 to T
N = 300; % there are going to be 300 steps
dt = T/N; % the discrete steps of the brownian



for ix = 1:nReps
    % based on the empirical data extracted
    % from stokes and lauffenburger
    beta = 0.99; % h^-1
    alpha = 1900; %µm^2 h-3
    % we define this parameter arbitrarialy
    % Now we generate the kappa parameter using the derivation of S and L
    % using the delta parameter
    delta = 3; % adimentional chemoatractant
    a0 = 10^-10; % initial concentration of the chemoatractant
    kappa = (alpha/beta) * (1/a0) * delta;

    %Fractional Brownian Path
    %Now we define a fractional browian motion


    %Now we plot the brownian walk for the x and y variables to see if they make sense
    % plot([0:dt:T],dW(1,:)), hold on
    % plot([0:dt:T],dW(2,:)), hold off

    %Begining of the modeling
    %To solve the stochastic differential equation model we define the following L interger and stepsize
    R = 2;
    Dt = R*dt; % We define the Dt as a multiple of dt
    L = N/R; % L is used to discretize the time
    % We initialize the solution vector
    Xem = zeros(2,L); % firsNt x
    % second y
    Vem = zeros(2,L);

    % Initialize the first point
    Xzero = [ix*1;0];
    Xtemp = Xzero;
    Vzero = [0;0];
    Vtemp = Vzero;
    % And we initialize the first angles
    theta = pi/2;
    phi = pi/2;

    dW = sqrt(dt) * randn(2,N);
    W = cumsum(dW);

   % These equations have been implemented in the functions that are in the folder functions

    % plot([1:100],da(:))
    % We beging the for loop for the model
    Theta = zeros(L);
    Theta(1) = theta;
    DA_list = zeros(2,L);
    Phi = zeros(L);
    Phi(1) = phi;
    j = 1;
    end_simu = false;
    for j = 1:L

        % simulation

        winc = sum(dW(:,R*(j-1)+1:R*j),2);
        DA = gradient_const(Xtemp(1,:),Xtemp(2,:), 8e-15);
        Vtemp = Vtemp + (-beta * Vtemp + kappa * DA * sin(abs(phi/2)))*Dt + sqrt(alpha)*winc;
        Vem(:,j) = Vtemp;
        Xtemp = Xtemp + Vtemp * Dt;

        % so that the y coord is not negative
        if Xtemp(2,:) < 0 && j == 1
            Xtemp = Xzero;
        elseif Xtemp(2,:) < 0 && j >= 2
            Xtemp = Xem(:,j-1);
        end
        Xem(:,j) = Xtemp;

        % calculation of the angles
        if j ~= 1
            xj = Xem(1,j);
            yj = Xem(2,j);
            xj_1 = Xem(1,j-1);
            yj_1 = Xem(2,j-1);
            theta = atan((yj-yj_1)/(xj-xj_1));
            phi = calculate_phi( xj, yj , xj, -100,0);
            Theta(j) = theta;
        end
        % storing of data
        DA_list(:,j) = DA;
        Phi(j) = phi;

        % we store the time in which the sprout touches the 200 mu m
        % and when it does we store the time in Time_possition
        %{
        if Time_possition(ix)==0 & Xem(2,j) > 200
            Time_possition(ix) = Dt*j;
            break

        end
        %}
        %if end_simu == false
        %5    j = j + 1;
        %end
    end
    % We see the array of solutions for the position


    tSpan = linspace(1, j*Dt, j);
    Xdata{ix} = [Xem(1,1:j);Xem(2,1:j);tSpan];
    DA_data{ix} = [DA_list(:,1:j); tSpan];
    Theta_data{ix} = [Theta(1:j);tSpan];
    Phi_data{ix} = [Phi(1:j); tSpan];
    V_data{ix} = [Vem(1,1:j); Vem(2,1:j); tSpan ];

end






%{
figure(4)
;
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
%}
%%

figure(1)

cm = jet(nReps);

for jx = 1:1000:nReps
    Xtemp = Xdata{jx};
    name = ['Reps =',num2str(jx)];
    plot(Xtemp(1,:),Xtemp(2,:),'Color', cm(jx,:),'LineStyle','-.','DisplayName',name, ...
        'LineWidth',2)
    hold on
end
axis tight
%legend show
xlabel('X position')
ylabel('Y position')
title('Angiogenesis based on Brownian Motion')
saveas(gcf,"angiogenesis.svg")
hold off
%%
figure(5)
for jx = 1:nReps
    Vemp = V_data{jx};
    name = ['Reps =',num2str(jx)];
    plot(Vemp(1,:),Vemp(2,:),'Color', cm(jx,:),'LineStyle','-.','DisplayName',name, ...
        'LineWidth',10);
    hold on
end
axis tight
%legend show
xlabel('X velocity')
ylabel('Y velocity')
title('Angiogenesis based on Brownian Motion')
%%
%{
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
%}

% now we plot the time in which each sprout touched the 400 mu m
%{
figure(6)

scatter(1:nReps, Time_possition, 'Marker','.','CData',cm, 'SizeData',3e2)
hold on
xlabel('Número de Repetición')
ylabel('Tiempo (h)')
title('Tiempo de llegada a 200 mu m')
%}

%% Save the time possition variable

% we eliminate the 0 values and store the quantity of them

time_0_bm = Time_possition(Time_possition ~= 0);
n_time_0_bm = length(Time_possition(Time_possition == 0));
Time_possition_bm = Time_possition;
% We export these values

% Generate Histogram
figure(7)

h = histogram(time_0_bm, 'BinEdges',[0:1:T]); hold on

leg = ['#0 value = ', num2str(n_time_0_bm)];

legend(leg)

removeToolbarExplorationButtons(h);


% save it

exportgraphics(gcf, file_name_hist ...
    ,'ContentType','vector')

%% generate the index for the different deciles




sorted_t0_bm = sort(time_0_bm);
division = int64(length(sorted_t0_bm)/10);
indexes = [];
for i = linspace(1, length(time_0_bm), 10)
    value = sorted_t0_bm(int64(i));
    index = find(Time_possition_bm == value, 1, "first");
    indexes = [indexes, index];
end 








% We plot the sprouts that behave in this ways in the 10 ways

figure(8)
cm = jet(length(indexes));
color = 1;
for jx = indexes

    X = Xdata{jx};
    name = ['Time =',num2str(Time_possition(jx))];
    x_movido = X(1,:) - (max(X(1,:)));
    plot(x_movido,X(2,:),'Color', cm(color,:),'LineStyle','-','DisplayName',name, ...
        'LineWidth',2);
    color = color + 1;
    hold on
end
axis tight
legend show
xlabel('X position')
ylabel('Y position')
title('Angiogenesis based on Brownian Motion')
% save sprouts

%%



