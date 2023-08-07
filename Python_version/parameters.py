# definition of parameters is now in parameters.py

T = 3
N = 2000

Beta = 1/3
alpha = 40 * 1/3
kappa = 10
R = 4
Dt = R * (T/N) # Dt = R * dt
L = int(N/R)
GRAD = 2e-15

if __name__ == "__main__":
    print(L)