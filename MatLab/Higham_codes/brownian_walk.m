%% Brownian path

randn('state',100) % select the state of the random numbers

T=1; N=500; dt = T/N;

% preallocation of the arrays for increase efficiency
dW = zeros(1,N); 
W = zeros(1,N);

%approximation outside the loop
dW(1) = sqrt(dt) *  randn;
W(1) = dW(1);

for j = 2:N
    dW(j) = sqrt(dt) * randn;
    W(j) = W(j-1) + dW(j);
end

plot([0:dt:T],[0,W],'r-');
xlabel('t','FontSize',16)
ylabel('W(t)','FontSize',16,'Rotation',0)