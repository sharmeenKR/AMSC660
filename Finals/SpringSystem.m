function SpringSystem()
    close all;
Rhoop = 3; % the radius of the hoop
r0 = 1; % the equilibrial length of the springs
kappa = 1; % the spring constant
Nnodes = 21;
A = zeros(Nnodes,Nnodes); % spring adjacency matrix
% vertical springs
for k = 1 : 3
    A(k,k+4) = 1;
end
for k = 5 : 7  
    A(k,k+5) = 1;
end
for k = 10 : 12  
    A(k,k+5) = 1;
end
for k = 15 : 17  
    A(k,k+4) = 1;
end
% horizontal springs
for k = 4 : 7
    A(k,k+1) = 1;
end
for k = 9 : 12  
    A(k,k+1) = 1;
end
for k = 14 : 17  
    A(k,k+1) = 1;
end
% symmetrize
Asymm = A + A';
%indices of nodes on the hoop
ind_hoop = [0,3,8,13,18,19,20,17,12,7,2,1] + 1;
Nhoop = length(ind_hoop);
% indices of free nodes (not attached to the hoop)
ind_free = [4,5,6,9,10,11,14,15,16] + 1;
Nfree = length(ind_free);
% list of springs
[springs_0,springs_1] = ind2sub([Nnodes,Nnodes],find(A));
springs = [springs_0';springs_1'];

Nsprings = length(springs_0);
% maps indices of nodes to their indices in the gradient vector

%% Initialization

% Initial angles for the nodes are uniformly distributed around the range of 2*pi
% startting from theta0 and going counterclockwise
theta0 = 2*pi/3;
theta = theta0 + linspace(0,2*pi,Nhoop+1)';
theta(end) = [];
% Initial positions
pos = zeros(Nnodes,2);
pos(ind_hoop,1) = Rhoop*cos(theta);
pos(ind_hoop,2) = Rhoop*sin(theta);
pos(ind_free,1) = [-1.,0.,1.,-1.,0.,1.,-1.,0.,1.]';
pos(ind_free,2) = [1.,1.,1.,0.,0.,0.,-1.,-1.,-1.]'; 

% Initiallize the vector of parameters to be optimized
vec = [theta;pos(ind_free,1);pos(ind_free,2)]; % a column vector with 30 components

% Results storage
results = struct();

%% Gauss-Newton Optimization
  vec_gn = vec;
  tol = 1e-6;
  max_iter = 100;
  iter = 0;
  gn_energy = [];
  gn_gradient_norm = [];

  while true
      grad = compute_gradient(vec_gn, Asymm, r0, kappa, Rhoop, ind_hoop, ind_free);
      J = compute_jacobian(vec_gn, Asymm, r0, kappa, Rhoop, ind_hoop, ind_free);

      % Gauss-Newton step
      delta_vec = -pinv(J' * J) * J' * grad;
      vec_gn = vec_gn + delta_vec;

      % Store energy and gradient norm
       gn_energy(end+1) = Energy(vec_gn, springs, r0, kappa, Rhoop, ind_hoop, ind_free);
       gn_gradient_norm(end+1) = norm(grad);

     % Convergence check
       if norm(delta_vec) < tol || iter >= max_iter
          break;
       end
       iter = iter + 1;
  end

% Store Gauss-Newton results
results.gn_energy = gn_energy;
results.gn_gradient_norm = gn_gradient_norm;
results.gn_vec = vec_gn;
fprintf('Gauss-Newton Minimal Spring Energy: %.6f\n', gn_energy(end));
[~, pos_gn] = vec_to_pos(vec_gn, Rhoop, ind_hoop, ind_free);
fprintf('Gauss-Newton Gradient Norm: %.6f\n', gn_gradient_norm(end));
fprintf('Gauss-Newton Final Node Positions:\n');
disp(pos_gn);

%% Adam Optimization
vec_a = vec;
tol = 1e-6;
max_iter = 100;
alpha = 0.1; % Learning rate
beta1 = 0.9; % Exponential decay rate for momentum
beta2 = 0.999; % Exponential decay rate for the second moment
epsilon = 1e-8; % Small value to avoid division by zero

m = zeros(size(vec_a)); % Initialize 1st moment vector
v = zeros(size(vec_a)); % Initialize 2nd moment vector
iter = 0;
a_energy = [];
a_gradient_norm = [];

while true
% Compute gradient
grad = compute_gradient(vec_a, Asymm, r0, kappa, Rhoop, ind_hoop, ind_free);

% Nesterov step (lookahead position)
m = beta1 * m + (1 - beta1) * grad; % Update biased first moment estimate
v = beta2 * v + (1 - beta2) * (grad .^ 2); % Update biased second moment estimate

m_hat = m / (1 - beta1^(iter + 1)); % Bias-corrected first moment estimate
v_hat = v / (1 - beta2^(iter + 1)); % Bias-corrected second moment estimate

% Update vector using Nesterov momentum and Adam
 vec_prev = vec_a;
 vec_a = vec_a - alpha * (m_hat ./ (sqrt(v_hat) + epsilon));

% Store energy and gradient norm
 a_energy(end+1) = Energy(vec_a, springs, r0, kappa, Rhoop, ind_hoop, ind_free);
 a_gradient_norm(end+1) = norm(grad);

% Convergence check
  if norm(vec_a - vec_prev) < tol || iter >= max_iter
       break;
  end
      iter = iter + 1;
end

% Store Adam results
results.a_energy = a_energy;
results.a_gradient_norm = a_gradient_norm;
results.a_vec = vec_a;
fprintf('Nesterov-Adam Minimal Spring Energy: %.6f\n', a_energy(end));
[~, pos_a] = vec_to_pos(vec_a, Rhoop, ind_hoop, ind_free);
fprintf('Adam Gradient Norm: %.6f\n', a_gradient_norm(end));
fprintf('Adam Final Node Positions:\n');
disp(pos_a);

%% Visualization
% Final positions for Gauss-Newton
draw_spring_system(pos_gn, springs, Rhoop, ind_hoop, ind_free);
title('Gauss-Newton Optimization');

% Final positions for Nesterov-Adam
draw_spring_system(pos_a, springs, Rhoop, ind_hoop, ind_free);
title('Adam Optimization');

% Plot energy vs iteration
figure;
subplot(2, 1, 1);
plot(1:length(gn_energy), gn_energy, '-o', 'DisplayName', 'Gauss-Newton Energy');
hold on;
plot(1:length(a_energy), a_energy, '-x', 'DisplayName', 'Adam Energy');
xlabel('Iteration');
ylabel('Spring Energy');
legend;
title('Spring Energy vs Iteration');
xlim([0 40])

% Plot gradient norm vs iteration
subplot(2, 1, 2);
plot(1:length(gn_gradient_norm), gn_gradient_norm, '-o', 'DisplayName', 'Gauss-Newton Gradient Norm');
hold on;
plot(1:length(a_gradient_norm), a_gradient_norm, '-x', 'DisplayName', 'Adam Gradient Norm');
xlabel('Iteration');
ylabel('Gradient Norm');
legend;
title('Gradient Norm vs Iteration');
xlim([0 40])
end


function J = compute_jacobian(vec, Asymm, r0, kappa, R, ind_hoop, ind_free)
epsilon = 1e-8;
n = length(vec);
J = zeros(n, n);
for i = 1:n
    vec_perturbed = vec;
    vec_perturbed(i) = vec_perturbed(i) + epsilon;
    grad_perturbed = compute_gradient(vec_perturbed, Asymm, r0, kappa, R, ind_hoop, ind_free);
    grad = compute_gradient(vec, Asymm, r0, kappa, R, ind_hoop, ind_free);
    J(:, i) = (grad_perturbed - grad) / epsilon;
end
end

%%
function draw_spring_system(pos,springs,R,ind_hoop,ind_free)
% draw the hoop 
figure;
hold on;
t = linspace(0,2*pi,200);
plot(R*cos(t),R*sin(t),'linewidth',5,'color','r');
% plot springs
Nsprings = size(springs,2);
for k = 1 : Nsprings
    j0 = springs(1,k);
    j1 = springs(2,k);
    plot([pos(j0,1),pos(j1,1)],[pos(j0,2),pos(j1,2)],'linewidth',3,'color','k');
end
% plot nodes
plot(pos(ind_hoop,1),pos(ind_hoop,2),'.','Markersize',100,'Color',[0.5,0,0]);
plot(pos(ind_free,1),pos(ind_free,2),'.','Markersize',100,'Color','k');
set(gca,'Fontsize',20);
daspect([1,1,1]);
end

%% 
function grad = compute_gradient(vec,Asymm,r0,kappa,R,ind_hoop,ind_free)
    [theta,pos] = vec_to_pos(vec,R,ind_hoop,ind_free);    
    Nhoop = length(ind_hoop);
    g_hoop = zeros(Nhoop,1); % gradient with respect to the angles of the hoop nodes
    Nfree = length(ind_free);
    g_free = zeros(Nfree,2); % gradient with respect to the x- and y-components of the free nodes
    for k = 1 : Nhoop
        ind = find(Asymm(ind_hoop(k),:)); % index of the node adjacent to the kth node on the hoop
        rvec = pos(ind_hoop(k),:) - pos(ind,:); % the vector from that adjacent node to the kth node on the hoop
        rvec_length = norm(rvec); % the length of this vector
        g_hoop(k) = (rvec_length - r0)*R*kappa*(rvec(1)*(-sin(theta(k))) + rvec(2)*cos(theta(k)))/rvec_length;
    end
    for k  = 1 : Nfree
        ind = find(Asymm(ind_free(k),:)); % indices of the nodes adjacent to the kth free node
        Nneib = length(ind);
        for j = 1 : Nneib
            rvec = pos(ind_free(k),:) - pos(ind(j),:); % the vector from the jth adjacent node to the kth free node 
            rvec_length = norm(rvec);  % the length of this vector
            g_free(k,:) = g_free(k,:) + (rvec_length - r0)*R*kappa*rvec/rvec_length;
        end
    end
    % return a single 1D vector
    grad = [g_hoop;g_free(:,1);g_free(:,2)];
end

%%
function E = Energy(vec,springs,r0,kappa,R,ind_hoop,ind_free)
    [~,pos] = vec_to_pos(vec,R,ind_hoop,ind_free);
    Nsprings = size(springs,2);
    E = 0.;
    for k =1 : Nsprings
        j0 = springs(1,k);
        j1 = springs(2,k);
        rvec = pos(j0,:) - pos(j1,:);
        rvec_length = norm(rvec);       
        E = E + kappa*(rvec_length - r0)^2;
    end
    E = E*0.5;
end

%%
function [theta,pos] = vec_to_pos(vec,R,ind_hoop,ind_free)
    Nhoop = length(ind_hoop);
    Nfree = length(ind_free);
    Nnodes = Nhoop + Nfree;
    theta = vec(1:Nhoop);
    pos = zeros(Nnodes,2);
    pos(ind_hoop,1) = R*cos(theta);
    pos(ind_hoop,2) = R*sin(theta);
    % positions of the free nodes
    pos(ind_free,1) = vec(Nhoop+1:Nnodes);
    pos(ind_free,2) = vec(Nnodes+1:end); 
end

