function GMM_EM( X )
%GMM Summary of this function goes here
%   Detailed explanation goes here

% N : the number of data instances
% D : dimension of each data instance
% X : N * D
% mu : D * 1
% sigma : (D * N) * (N * D) = D * D

D = size(X, 2); 
N = size(X, 1);
K = 2               % the number of clusters

% initial params
Params.mu = [-1 1; 1 -1]';
Params.sigma = repmat(eye(K), [1 1 K]);
Params.pi = normalize(ones(1,K));
Params.gamma = zeros(N,K);

while 1
    %% Expectation step
    weighted_likelihoods = zeros(N,K);
    for k = 1:K
       for i = 1:N
           % Here, we have to calculate p(x_i|theta_k), not log(p(x_i|theta_k))
           % Validation required
           weighted_likelihoods(i,k) = Params.pi(:,k)*exp(likelihood(X(i,:), Params.mu(:,k), Params.sigma(:,:,k)));
       end
    end
    
    for k = 1:K
       for i = 1:N
          Params.gamma(i,k)  = weighted_likelihoods(i,k) / sum(weighted_likelihoods(:,k));
       end
    end
    
    %% Maximization step
    for k = 1:K
        gamma_k = sum(Params.gamma(:,k));
        
        % new pi
        Params.pi(k) = gamma_k / N;
        
        % new mu
        Params.mu(:,k) = (Params.gamma(:,k)'*X) / gamma_k;
        
        % new sigma
        mu_k = Params.mu(:,k);
        weighted_scatter_mat = zeros(D);
        for i = 1:N
            weighted_scatter_mat = weighted_scatter_mat + Params.gamma(i,k)*Params.gamma(i,:)'*Params.gamma(i,:);
        end
        Params.sigma(:,:,k) = (weighted_scatter_mat / gamma_k) - (mu_k*mu_k');
    end
    
    Q = complete_log_likelihood(X, Params)
end
end

%% Likelihood
% log(p(x_i|theta_k)) vs. p(x_i|theta_k)
function L = likelihood(x_i, mu_k, sigma_k)
    L = log(det(sigma_k)) + (x_i' - mu_k)' * inv(sigma_k) * (x_i' - mu_k);
end

%% Complete Data Log Likelihood
function Q = complete_log_likelihood(X, Params)
    Q = 0;
    K = size(Params.pi, 2);
    N = size(X, 1);

    for k = 1:K
       for i = 1:N
          Q = Q + (Params.gamma(i,k)*log(Params.pi(:,k)) + (Params.gamma(i,k)*exp(likelihood(X(i,:), Params.mu(:,k), Params.sigma(:,:,k))))); 
       end
    end
end
