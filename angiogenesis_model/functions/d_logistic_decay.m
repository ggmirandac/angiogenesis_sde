function [dld] = d_logistic_decay(L,b,c,y)
    num = L*b*c*exp(-c*y);
    den = (1+b*exp(-c*y))^2;
    dld=num/den;
end