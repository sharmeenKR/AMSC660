function mnist_lda()
    close all;
    mdata = load('mnist.mat');
    imgs_train = mdata.imgs_train;
    labels_train = mdata.labels_train;

    %% Find 3, 8, and 9 in training data
    ind3 = find(double(labels_train) == 3);
    ind8 = find(double(labels_train) == 8);
    ind9 = find(double(labels_train) == 9);
    train3 = imgs_train(:, :, ind3);
    train8 = imgs_train(:, :, ind8);
    train9 = imgs_train(:, :, ind9);
    img_size = size(train3, 1) * size(train3, 2);
    X3 = reshape(train3, img_size, []).';
    X8 = reshape(train8, img_size, []).';
    X9 = reshape(train9, img_size, []).';
    Xtrain = [X3; X8; X9];
    label = [ones(size(X3, 1), 1) * 3; ...
             ones(size(X8, 1), 1) * 8; ...
             ones(size(X9, 1), 1) * 9];

    [U, ~, ~] = svd(Xtrain', 'econ');
    nPCA = 20; % Number of principal components
    Xtrain_pca = Xtrain * U(:, 1:nPCA);
    m = mean(Xtrain_pca, 1);

    %% Scatter matrices
    fprintf("Computing scatter matrices...\n");
    categories = unique(label);
    S_w = zeros(nPCA, nPCA);
    S_b = zeros(nPCA, nPCA);

    for i = 1:length(categories)
        class_data = Xtrain_pca(label == categories(i), :);
        n_i = size(class_data, 1);
        m_i = mean(class_data, 1);
        S_w = S_w + (class_data - m_i)' * (class_data - m_i);
        S_b = S_b + n_i * (m_i - m)' * (m_i - m);
    end

    %% Solve the generalized eigenvalue problem
    fprintf("Solving the generalized eigenvalue problem...\n");
    [V, D] = eig(S_b, S_w);
    [~, sorted_indices] = sort(diag(D), 'descend');
    W = V(:, sorted_indices(1:2));

    %% Project data onto the LDA space
    mapped_data = Xtrain_pca * W;
    figure;
    hold on;
    colors = {'r', 'g', 'b'}; % Red for 3, Green for 8, Blue for 9
    for i = 1:length(categories)
        scatter(mapped_data(label == categories(i), 1), mapped_data(label == categories(i), 2), 50, colors{i}, 'filled');
    end
    legend({'3', '8', '9'}, 'Location', 'Best');
    title('LDA Projection of MNIST Data');
    grid on;
    hold off;
end
