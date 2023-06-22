function [dld] = d_logistic_decay(a,b,c,y)
    num = a*b*c*exp(-c*y);
    den = (1+b*exp(-c*y))^2;
    dld=num/den;
end