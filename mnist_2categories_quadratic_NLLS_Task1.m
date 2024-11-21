function mnist_2categories_quadratic_NLLS_Task1()
close all
fsz = 20;
mdata = load('mnist.mat');
imgs_train = mdata.imgs_train;
imgs_test = mdata.imgs_test;
labels_test = mdata.labels_test;
labels_train = mdata.labels_train;
%% find 1 and 7 in training data
ind1 = find(double(labels_train)==2);
ind2 = find(double(labels_train)==8);
n1train = length(ind1);
n2train = length(ind2);
fprintf("There are %d 1's and %d 7's in training data\n",n1train,n2train);
train1 = imgs_train(:,:,ind1);
train2 = imgs_train(:,:,ind2);
fprintf("Size of the images: %d - by - %d\n",size(train1,1),size(train1,2));
%% find 1 and 7 in test data
itest1 = find(double(labels_test)==2);
itest2 = find(double(labels_test)==8);
n1test = length(itest1);
n2test = length(itest2);
fprintf("There are %d 1's and %d 7's in test data\n",n1test,n2test);
test1 = imgs_test(:,:,itest1);
test2 = imgs_test(:,:,itest2);
%% plot some data from category 1
figure; colormap gray
for j = 1:20
    subplot(4,5,j);
    imagesc(train1(:,:,j));
end
%% plot some data from category 2
figure; colormap gray
for j = 1:20
    subplot(4,5,j);
    imagesc(train2(:,:,j));
end

    %% Prepare training data for PCA
    [d1, d2, ~] = size(train1);
    X1 = reshape(train1, d1 * d2, n1train)';
    X2 = reshape(train2, d1 * d2, n2train)';
    X = [X1; X2];
    labels = [ones(n1train, 1); -ones(n2train, 1)];

    %% Perform SVD for PCA
    [U, ~, ~] = svd(X', 'econ');

    %% Loop over different numbers of PCA components
    pca_components = [1, 5, 10, 15, 20];
    misclassified_counts = zeros(length(pca_components), 1);

    for i = 1:length(pca_components)
        nPCA = pca_components(i);
        Xpca_train = X * U(:, 1:nPCA);

        %% Prepare test data
        Xtest1 = reshape(test1, d1 * d2, n1test)';
        Xtest2 = reshape(test2, d1 * d2, n2test)';
        Xtest = [Xtest1; Xtest2] * U(:, 1:nPCA);
        test_labels = [ones(n1test, 1); -ones(n2test, 1)];

        %% Solve quadratic classification problem
        w = ones(nPCA^2 + nPCA + 1, 1);
        r_and_J = @(w) Res_and_Jac(Xpca_train, labels, w);
        [w, ~, ~] = LevenbergMarquardt(r_and_J, w, 600, 1e-3);

        %% Evaluate on test data
        predictions = myquadratic(Xtest, test_labels, w);
        misclassified_counts(i) = sum(predictions < 0);
    end

    %% Plot results
    figure;
    plot(pca_components, misclassified_counts, '-o', 'LineWidth', 2);
    xlabel('Number of PCAs', 'FontSize', fsz);
    ylabel('Number of Misclassified Digits', 'FontSize', fsz);
    title('Effect of PCAs on Classification Accuracy', 'FontSize', fsz);
    grid on;
end
function f = fun0(X,y,w)
f = 0.5*sum((log(1 + exp(-myquadratic(X,y,w)))).^2);
end
%%
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


