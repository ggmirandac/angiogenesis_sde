function [ld] = logistic_decay(L,b,c,y)
    ld = (L/(1+b*exp(-c*y)));
end