import math

def phi_calc(xi, yi, xa, ya, theta):
    num = (xi-xa)*math.cos(theta)+(yi-ya)*math.sin(theta)
    den = math.sqrt((xi-xa)**2+(yi-ya)**2)
    ang_phi = math.acos(num/den)
    return ang_phi


