function calculate_phi(xi,yi,xa,ya,theta)

    num = (xi-xa)*cos(theta) + (yi-ya)*sin(theta)
    den = sqrt((xi-xa)^2 + (yi-ya)^2)
    phi = acos(num/den)
    return phi
end
