function mnist_2categories_quadratic_NLLS_Task2()
    close all;
    fsz = 20;
    mdata = load('mnist.mat');
    imgs_train = mdata.imgs_train;
    imgs_test = mdata.imgs_test;
    labels_test = mdata.labels_test;
    labels_train = mdata.labels_train;

    %% Process training and test data for digits 1 and 7
    ind1 = find(double(labels_train) == 2);
    ind2 = find(double(labels_train) == 8);
    n1train = length(ind1);
    n2train = length(ind2);
    train1 = imgs_train(:, :, ind1);
    train2 = imgs_train(:, :, ind2);

    itest1 = find(double(labels_test) == 2);
    itest2 = find(double(labels_test) == 8);
    n1test = length(itest1);
    n2test = length(itest2);
    test1 = imgs_test(:, :, itest1);
    test2 = imgs_test(:, :, itest2);

    %% Prepare training data for PCA
    [d1, d2, ~] = size(train1);
    X1 = reshape(train1, d1 * d2, n1train)';
    X2 = reshape(train2, d1 * d2, n2train)';
    X = [X1; X2];
    labels = [ones(n1train, 1); -ones(n2train, 1)];

    %% Perform SVD for PCA
    [U, ~, ~] = svd(X', 'econ');
    nPCA = 20; % Fix the number of PCA components
    Xpca_train = X * U(:, 1:nPCA);

    %% Define Gauss-Newton Algorithm
    w = ones(nPCA^2 + nPCA + 1, 1); % Initialize weights
    max_iter = 200;
    tol = 1e-3;

    % Storage for plotting
    loss_values = zeros(max_iter, 1);
    grad_norm_values = zeros(max_iter, 1);

    for iter = 1:max_iter
        % Calculate residual and Jacobian
        [r, J] = Res_and_Jac(Xpca_train, labels, w);

        % Compute loss function
        loss = F(r);
        loss_values(iter) = loss;

        % Compute gradient and its norm
        g = J' * r;
        grad_norm = norm(g);
        grad_norm_values(iter) = grad_norm;

        % Check for convergence
        if grad_norm < tol
            fprintf('Converged at iteration %d\n', iter);
            break;
        end

        % Compute Gauss-Newton update step with regularization
        H = J' * J + eye(length(w)) * 1e-6; % Regularized Hessian
        p = -H \ g; % Update step
        w = w + p; % Update weights
    end

    %% Plot Loss and Gradient Norm
    figure;
    plot(1:iter, loss_values(1:iter), 'LineWidth', 2);
    xlabel('Iteration', 'FontSize', fsz);
    ylabel('Loss (f)', 'FontSize', fsz);
    title('Loss vs Iterations (Gauss-Newton)', 'FontSize', fsz);
    set(gca,'fontsize',fsz,'Yscale','log');
    grid on;

    figure;
    plot(1:iter, grad_norm_values(1:iter), 'LineWidth', 2);
    xlabel('Iteration', 'FontSize', fsz);
    ylabel('Norm of Gradient (||g||)', 'FontSize', fsz);
    title('Gradient Norm vs Iterations (Gauss-Newton)', 'FontSize', fsz);
    set(gca,'fontsize',fsz,'Yscale','log');
    grid on;
end

%% Helper Functions
function f = F(r)
    % Compute the loss function value
    f = 0.5 * sum(r.^2);
end

function [r,J] = Res_and_Jac(X,y,w)
% vector of residuals
aux = exp(-myquadratic(X,y,w));
r = log(1 + aux);
% the Jacobian matrix
a = -aux./(1+aux);
[n,d] = size(X);
d2 = d^2;
ya = y.*a;
qterm = zeros(n,d2);
for k = 1 : n
    xk = X(k,:); % row vector x
    xx = xk'*xk;
    qterm(k,:) = xx(:)';
end
Y = [qterm,X,ones(n,1)];
J = (ya*ones(1,d2+d+1)).*Y;
end
%%
function q = myquadratic(X,y,w)
d = size(X,2);
d2 = d^2;
W = reshape(w(1:d2),[d,d]);
v = w(d2+1:d2+d);
b = w(end);
qterm = diag(X*W*X');
q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end
