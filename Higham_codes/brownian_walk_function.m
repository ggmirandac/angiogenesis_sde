% Function along a Brownian path

randn('state', 100)

T = 1; N = 500; dt = T/N; t = [dt:dt:1]; % t is the time

M = 1000; % we generate 1000 paths

dW = sqrt(dt) * randn(M,N); % generate a dW of size M rows and N columns

W = cumsum(dW,2); % 

U = exp(repmat(t,[M 1]) + 0.5*W);

Umean = mean(U);

plot([0,t],[1,Umean],'b-'), hold on
plot([0,t],[ones(5,1),U(1:5,:)],'r--'),hold off

xlabel('t','FontSize',16)
ylabel('U(t)','FontSize',16,'Rotation',0,'HorizontalAlignment','right')
legend('meand of 1000 paths', '5 individual paths')