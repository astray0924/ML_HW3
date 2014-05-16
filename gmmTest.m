% N : the number of data
% D : dimension
% X : N * D
% mu : D * 1
% sigma : (D * N) * (N * D) = D * D
data = load('oldfaithful.mat');
X = data.dataset;

% Expectation Maximization for Gaussian Mixture Models
GMM_EM(X);


