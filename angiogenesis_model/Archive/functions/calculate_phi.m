function [ang_phi] = calculate_phi(xi,yi, xa, ya, theta)
%  phi calculate the angle phi, which is the angle 
%  between the point of the sprout and the location of the attractant
num = (xa-xi)*cos(theta) + (ya-yi)*sin(theta);
den = ((xa-xi)^2+(ya-yi)^2)^(1/2);
ang_phi = acos(num/den);
end