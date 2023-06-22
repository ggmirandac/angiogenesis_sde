function [ld] = logistic_decay(a,b,c,y)
    ld = (a/(1+b*exp(-c*y)));
end