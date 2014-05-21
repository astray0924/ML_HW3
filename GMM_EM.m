function Result = GMM_EM( X )
%GMM Summary of this function goes here
%   Detailed explanation goes here

% N : the number of data instances
% D : dimension of each data instance
% X : N * D
% mu : D * 1
% sigma : (D * N) * (N * D) = D * D

D = size(X, 2); 
N = size(X, 1);
K = 2;               % the number of clusters

% initial params
Params.mu = [-1 1; 1 -1]';
Params.sigma = repmat(eye(K), [1 1 K]);
Params.pi = [0.5 0.5];
Params.gamma = ones(N,K);
Params.gamma_prev = Params.gamma;

% Result
Result.Q = 0;
Result.Q_prev = 0;

% Q history
global historyQ;
historyQ = zeros;

while 1
    %% Expectation step
    % save the current gamma as gamma_prev
    Params.gamma_prev = Params.gamma;
    
    % calculate new gamma
    weighted_likelihoods = zeros(N,K);
    for k = 1:K
       for i = 1:N
           weighted_likelihoods(i,k) = Params.pi(:,k)*exp(log_likelihood(X(i,:), Params.mu(:,k), Params.sigma(:,:,k)));
       end
    end
    
    for k = 1:K
       for i = 1:N
          Params.gamma(i,k)  = weighted_likelihoods(i,k) / sum(weighted_likelihoods(i,:));
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
        weighted_scatter_mat = zeros(D);
        for i = 1:N
            weighted_scatter_mat = weighted_scatter_mat + Params.gamma(i,k)*X(i,:)'*X(i,:);
        end
        mu_k = Params.mu(:,k);
        Params.sigma(:,:,k) = (weighted_scatter_mat / gamma_k) - (mu_k*mu_k');
    end
    
    % Result
    Result.Q_prev = Result.Q;
    Result.Q = complete_log_likelihood(X, Params);
    
    % record history of Q
    historyQ = [historyQ Result.Q];
    
    % Check terminate condition
    threshold = 0.0001e+04;
    if (Result.Q - Result.Q_prev) < threshold
       Result.Params = Params;
       
       % draw a Q graph
       % draw_Q_graph();
       
       % draw a result graph
       draw_result_graph(X, Params);
       
       break; 
    end
    
end
end

%% Draw Graph for Q
function draw_Q_graph()
    global historyQ;
    
    figure;
    
    x = 1:size(historyQ, 2);
    y = historyQ;
    
    plot(x, y, '--or');
end

%% Draw Graph for Result
function draw_result_graph(X, Params)
    N = size(X, 1);

    % create new window for the graph
    figure;

    x = X(:,1);
    y = X(:,2);
    colors = zeros(N,3);
    
    % fill color matrix
    for i = 1:N
       colors(i,:) = [1*Params.gamma(i,2) 0 1*Params.gamma(i,1)];
    end

    % draw graph
    for i = 1:N
        scatter(x, y, 10, colors);
    end
end

%% Likelihood
function L = log_likelihood(x_i, mu_k, sigma_k)
    L = log(det(sigma_k)) + (x_i' - mu_k)' * inv(sigma_k) * (x_i' - mu_k);
end

%% Complete Data Log Likelihood
function Q = complete_log_likelihood(X, Params)
    Q = 0;
    K = size(Params.gamma, 2);
    N = size(X, 1);

    for k = 1:K
       for i = 1:N
          Q = Q + (Params.gamma_prev(i,k)*log(Params.pi(:,k)) + (Params.gamma_prev(i,k)*log_likelihood(X(i,:), Params.mu(:,k), Params.sigma(:,:,k)))); 
       end
    end
end
