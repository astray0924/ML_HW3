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
K = 2;               % the number of cluster

% params
mu1 = [-1 1];
sigma1 = eye(K);
pi1 = 0.5;
resp1 = zeros(1,N);
resp1_prev = resp1;

mu2 = [1 -1];
sigma2 = eye(K);
pi2 = 0.5;
resp2 = zeros(1,N);
resp2_prev = resp2;

% Auxiliary value
Q = 0;
Q_history = [];

% Counter
counter = 0;

while 1
    counter = counter + 1;
    
    %% Expectation step    
    resp1_prev = resp1;
    resp2_prev = resp2;
    
    for i = 1:N
       x = X(i,:);
       norm = pi1*mvnpdf(x, mu1, sigma1) + pi2*mvnpdf(x, mu2, sigma2);
       resp1(1,i) = (pi1*mvnpdf(x, mu1, sigma1)) / norm;
       resp2(1,i) = (pi2*mvnpdf(x, mu2, sigma2)) / norm;
    end
%     resp1
    
    %% Maximization step
    % new pi
    pi1 = sum(resp1) / N;
    pi2 = sum(resp2) / N;
%     pi1
   
    % new mu
    mu1 = (resp1*X) / sum(resp1);
    mu2 = (resp2*X) / sum(resp2);
    
    % new covariance matrix
    wsm1 = zeros(D);
    wsm2 = zeros(D);
    for i = 1:N
        x = X(i,:);
        wsm1 = wsm1 + resp1(1,i)*x'*x;
        wsm2 = wsm2 + resp2(1,i)*x'*x;
    end
    sigma1 = (wsm1 / sum(resp1)) - (mu1'*mu1);
    sigma2 = (wsm2 / sum(resp2)) - (mu2'*mu2);
    sigma1 = (sigma1+sigma1')/2;
    sigma2 = (sigma2+sigma2')/2;
%     sigma2
        
    %% calculate Q
    Q = 0;
    for i = 1:N
       x = X(i,:);
       Q = Q + (resp1(1,i)*log(pi1) + resp2(1,i)*log(pi2)) + (resp1(1,i)*log(mvnpdf(x,mu1,sigma1)) + resp2(1,i)*log(mvnpdf(x,mu2,sigma2)));
    end
    
    Q_history = [Q_history Q];
%     Q_history
    
%     if counter > 17
%         draw_Q_graph(Q_history);
%         break;
%     end
    
%     draw_result_graph(X, resp1, resp2);

%     draw_result_graph(X, resp1, resp2);
    threshold = 1e-4;
    if counter > 1 &&  Q_history(end) - Q_history(end-1) < threshold
       counter
       
       % draw a Q graph
       draw_Q_graph(Q_history);
       
       % draw a result graph
       draw_result_graph(X, resp1, resp2);
       
       break; 
    end
end

end

%% Draw Graph for Q
function draw_Q_graph(Q_history)    
    figure;
    
    x = 1:size(Q_history, 2);
    y = Q_history;
    
    plot(x, y, '--or');
end

%% Draw Graph for Result
function draw_result_graph(X, resp1, resp2)
    N = size(X,1);
    
    figure;
    
    x = X(:,1);
    y = X(:,2);
    colors = zeros(N,3);
    
    % fill color matrix
    for i = 1:N
       colors(i,:) = [1*resp1(1,i) 0 1*resp2(1,i)];
    end
    
    % draw graph
    for i = 1:N
        scatter(x, y, 10, colors);
    end
end


