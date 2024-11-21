function mnist_2categories_quadratic_SGD_Task3()
    close all;
    fsz = 16;
    
    % Load and prepare data
    mdata = load('mnist.mat');
    imgs_train = mdata.imgs_train;
    labels_train = mdata.labels_train;

    % Process data for digits 2 and 8
    ind1 = find(double(labels_train) == 2);
    ind2 = find(double(labels_train) == 8);
    train1 = imgs_train(:, :, ind1);
    train2 = imgs_train(:, :, ind2);
    
    [d1, d2, n1train] = size(train1);
    n2train = size(train2, 3);
    
    % Prepare data for PCA
    X1 = reshape(train1, d1 * d2, n1train)';
    X2 = reshape(train2, d1 * d2, n2train)';
    X = [X1; X2];
    labels = [ones(n1train, 1); -ones(n2train, 1)];
    
    % PCA transformation
    [U, ~, ~] = svd(X', 'econ');
    nPCA = 20;
    Xpca_train = X * U(:, 1:nPCA);
    
    % SGD Parameters based on analysis
    max_epochs = 40;
    batch_sizes = [32, 64, 128];
    learning_rates = [0.01, 0.001, 0.0001];
    lambda = 0.01; % Regularization parameter
    
    % Store results for plotting
    loss_values = cell(length(batch_sizes), length(learning_rates));
    grad_norm_values = cell(length(batch_sizes), length(learning_rates));
    
    % Main loop for different configurations
    for b = 1:length(batch_sizes)
        batch_size = batch_sizes(b);
        for l = 1:length(learning_rates)
            lr = learning_rates(l);
            
            % Initialize weights
            w = ones(nPCA^2 + nPCA + 1, 1);
            
            % Training variables
            n_samples = size(Xpca_train, 1);
            n_batches = floor(n_samples/batch_size);
            current_loss_values = zeros(max_epochs * n_batches, 1);
            current_grad_norm_values = zeros(max_epochs * n_batches, 1);
            
            iter = 1;
            for epoch = 1:max_epochs
                % Shuffle data
                idx = randperm(n_samples);
                X_shuffled = Xpca_train(idx, :);
                y_shuffled = labels(idx);
                
                % Decreasing learning rate strategy
                current_lr = lr / sqrt(epoch);
                
                for batch = 1:n_batches
                    % Get batch
                    batch_idx = (batch-1)*batch_size + 1:min(batch*batch_size, n_samples);
                    X_batch = X_shuffled(batch_idx, :);
                    y_batch = y_shuffled(batch_idx);
                    
                    % Compute loss and gradient
                    [loss, grad] = compute_loss_and_gradient(X_batch, y_batch, w, lambda, length(batch_idx));
                    grad_norm = norm(grad);
                    
                    % Store metrics
                    current_loss_values(iter) = loss;
                    current_grad_norm_values(iter) = grad_norm;
                    
                    % Update weights
                    w = w - current_lr * grad;
                    
                    iter = iter + 1;
                end
            end
            
            % Store results
            loss_values{b,l} = current_loss_values(1:iter-1);
            grad_norm_values{b,l} = current_grad_norm_values(1:iter-1);
        end
    end
    
    % Plotting results
    plot_results(loss_values, grad_norm_values, batch_sizes, learning_rates, fsz);
end

function [loss, grad] = compute_loss_and_gradient(X, y, w, lambda, batch_size)
    % Compute quadratic terms
    q = myquadratic(X, y, w);
    
    % Compute loss according to equation (4)
    loss_term = (1/batch_size) * sum(log(1 + exp(-q)));
    reg_term = (lambda/2) * norm(w)^2;
    loss = loss_term + reg_term;
    
    % Compute gradient
    aux = exp(-q);
    grad_factor = -(1/batch_size) * (aux./(1 + aux));
    
    % Compute Jacobian
    [n, d] = size(X);
    d2 = d^2;
    qterm = zeros(n, d2);
    for k = 1:n
        xk = X(k,:);
        xx = xk'*xk;
        qterm(k,:) = xx(:)';
    end
    Y = [qterm, X, ones(n,1)];
    J = (y*ones(1,d2+d+1)).*Y;
    
    % Combine gradients
    grad_loss = J' * grad_factor;
    grad_reg = lambda * w;
    grad = grad_loss + grad_reg;
end

function q = myquadratic(X, y, w)
    d = size(X,2);
    d2 = d^2;
    W = reshape(w(1:d2),[d,d]);
    v = w(d2+1:d2+d);
    b = w(end);
    qterm = diag(X*W*X');
    q = y.*qterm + ((y*ones(1,d)).*X)*v + y*b;
end

function plot_results(loss_values, grad_norm_values, batch_sizes, learning_rates, fsz)
    colors = {'b', 'r', 'g'};
    figure('Position', [100, 100, 1200, 800]);
    
    for i = 1:length(batch_sizes)
        % Plot Loss
        subplot(3,2,2*i-1)
        for j = 1:length(learning_rates)
            plot(loss_values{i,j}, colors{j}, 'LineWidth', 2, ...
                 'DisplayName', sprintf('lr=%.4f', learning_rates(j)));
            hold on;
        end
        xlabel('Iterations', 'FontSize', fsz);
        ylabel('Loss', 'FontSize', fsz);
        title(sprintf('Loss vs Iterations (Batch Size=%d)', batch_sizes(i)), ...
              'FontSize', fsz);
        set(gca, 'YScale', 'log', 'FontSize', fsz);
        grid on;
        legend('show', 'Location', 'best');
        
        % Plot Gradient Norm
        subplot(3,2,2*i)
        for j = 1:length(learning_rates)
            plot(grad_norm_values{i,j}, colors{j}, 'LineWidth', 2, ...
                 'DisplayName', sprintf('lr=%.4f', learning_rates(j)));
            hold on;
        end
        xlabel('Iterations', 'FontSize', fsz);
        ylabel('Gradient Norm', 'FontSize', fsz);
        title(sprintf('Gradient Norm vs Iterations (Batch Size=%d)', ...
              batch_sizes(i)), 'FontSize', fsz);
        set(gca, 'YScale', 'log', 'FontSize', fsz);
        grid on;
        legend('show', 'Location', 'best');
    end
    
    % Add overall title
    subtitle('SGD Training Results with Different Configurations');
end
