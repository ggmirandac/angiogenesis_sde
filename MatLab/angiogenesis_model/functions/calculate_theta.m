function ang_theta = calculate_theta(xi, yi, xi_1, yi_1)
    m = (yi-yi_1)/(xi-xi_1);
    ang_theta = atan(m);
    if ang_theta < 0
        ang_theta = ang_theta + pi;
    end
        
end