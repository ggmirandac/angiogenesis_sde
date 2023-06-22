clc, close all
addpath("./functions/")
randn('state',100)

xtemp = [2;0]
camino = []
for i = 1:100
    valor_random = randi([-10,10],1);
    xtemp(2) = xtemp(2) + valor_random;
    xtemp

    camino(i) = gradient_div(xtemp(1),xtemp(2))
end

plot(1:100, camino)