function gradient = gradient_const(x, y, value)
%
%{
This code generates the gradient of the chemoatractant
in this case is a constant gradient on the y axis
inputs: 
    x: value of the x-coordinate
    y: value of the y-coordinate
    value: value of the constant gradient
outputs:
    grad: value of the gradient in the point (x,y)
%}
   if y >= 0
       grad = value;
   elseif y < 0
       grad = 0;
   end 

   gradient = [0; grad];
      
end