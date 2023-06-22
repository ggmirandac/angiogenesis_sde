
% Define the parameters
D = 0.5; % Diffusion coefficient
r_max = 1; % Maximum radial distance
t_max = 1; % Maximum time
N_r = 100; % Number of radial grid points
N_t = 500; % Number of time grid points

% Define the grid
r = linspace(0, r_max, N_r);
t = linspace(0, t_max, N_t);
[C, R] = meshgrid(r, r);

% Define the initial condition
C0 = zeros(N_r, 1);
C0(1) = 1;

% Solve the diffusion equation using the finite difference method
C_old = C0;
C_new = C_old;
dr = r(2) - r(1);
dt = t(2) - t(1);
for i = 1:N_t
    for j = 2:N_r-1
        C_new(j) = C_old(j) + D*dt/(dr^2*R(j)^2) * ((R(j)+dr/2)^2*(C_old(j+1)-C_old(j)) - (R(j)-dr/2)^2*(C_old(j)-C_old(j-1)));
    end
    C_new(1) = C_old(1) + 2*D*dt/(dr^2) * (C_old(2)-C_old(1));
    C_new(N_r) = C_old(N_r) + 2*D*dt/(dr^2*R(N_r)^2) * (C_old(N_r-1)-C_old(N_r));
    C_old = C_new;
end

% Remove duplicate values from R and corresponding values from C_new
[R_unique, idx] = unique(reshape(R,[],1));
C_new_unique = C_new(idx);

% Reshape R_unique to match the size of R
R_unique = reshape(R_unique, [numel(r), numel(r)]);

% Plot the heatmap of the concentration distribution
figure;
heatmap(R_unique, R_unique, C_new_unique);
xlabel('Radial distance');
ylabel('Radial distance');
title('Concentration distribution');
