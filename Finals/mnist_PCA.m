close all;
fsz = 20;
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

%% Center the data matrix
m = mean(Xtrain, 1); 
X_centered = Xtrain - m; 
[U, S, V] = svd(X_centered, 'econ'); 
W_pca = V(:, 1:2);
mapped_data_pca = X_centered * W_pca; % Project data onto the first two PCs

%% Plot the PCA results
figure;
hold on;
colors = {'r', 'g', 'b'}; % Red for 3, Green for 8, Blue for 9
categories = unique(label);
for i = 1:length(categories)
    scatter(mapped_data_pca(label == categories(i), 1), ...
            mapped_data_pca(label == categories(i), 2), ...
            50, colors{i}, 'filled');
end
legend({'3', '8', '9'}, 'Location', 'Best');
title('PCA Projection of MNIST Data (3, 8, 9)');
grid on;
hold off;
