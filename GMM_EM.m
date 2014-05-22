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

% Covariance mode
% Full          - 0
% Diagonal      - 1
% Spherical     - 2
global covMode;
covMode = 1;

% params
mu1 = [-1 1];
sigma1 = eye(K);
pi1 = 0.5;
resp1 = ones(1,N);
resp1_prev = resp1;

mu2 = [1 -1];
sigma2 = eye(K);
pi2 = 0.5;
resp2 = ones(1,N);
resp2_prev = resp2;

% Auxiliary value
Q = 0;
Q_history = [];

% Counter
iteration = 1;

% draw initial graph
draw_result_graph(X, resp1, resp2, iteration);

while 1
    iteration = iteration + 1;
    
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
    if covMode == 0 % Full
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
    elseif covMode == 1 % Diagonal
        temp1 = (X - repmat(mu1, N, 1)).^2;
        temp2 = (X - repmat(mu2, N, 1)).^2;
        
        class1_1 = resp1*temp1(:,1)/sum(resp1);
        class1_2 = resp1*temp1(:,2)/sum(resp1);
        class2_1 = resp2*temp2(:,1)/sum(resp2);
        class2_2 = resp2*temp2(:,2)/sum(resp2);
        
        sigma1 = [class1_1 0; 0 class1_2];
        sigma2 = [class2_1 0; 0 class2_2];
    elseif covMode == 2 % Spherical 
        temp1 = 0;
        temp2 = 0;
        
        for i = 1:N
           x = X(i,:);
           temp1 = temp1 + resp1(1,i)*(x - mu1)*(x - mu1)';
           temp2 = temp2 + resp2(1,i)*(x - mu2)*(x - mu2)';
        end
        
        temp1 = temp1/(D*sum(resp1));
        temp2 = temp2/(D*sum(resp2));
        
        sigma1 = temp1*eye(D);
        sigma2 = temp2*eye(D);
    end
        
    %% calculate Q
    Q = 0;
    for i = 1:N
       x = X(i,:);
       Q = Q + (resp1(1,i)*log(pi1) + resp2(1,i)*log(pi2)) + (resp1(1,i)*log(mvnpdf(x,mu1,sigma1)) + resp2(1,i)*log(mvnpdf(x,mu2,sigma2)));
    end
    
    Q_history = [Q_history Q];
    
    % draw graph
    if iteration == 3 || iteration == 5 || iteration == 16
        draw_result_graph(X, resp1, resp2, iteration);
    end

    threshold = 1e-4;
    if iteration > 2 &&  Q_history(end) - Q_history(end-1) < threshold
%        counter
       
       % draw a Q graph
       draw_Q_graph(Q_history);
       
       % draw a result graph
       draw_result_graph(X, resp1, resp2, iteration);
       
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
function draw_result_graph(X, resp1, resp2, iteration)
    N = size(X,1);
    
    global covMode;
    if covMode == 0
        name = 'Default';
    elseif covMode == 1
        name = 'Diagonal';
    elseif covMode == 2
        name = 'Spherical';
    end
    
    figure('Name', name);
    
    x = X(:,1);
    y = X(:,2);
    colors = zeros(N,3);
    
    % fill color matrix
    for i = 1:N
       colors(i,:) = [1*resp1(1,i) 0 1*resp2(1,i)];
    end
    
    % data points for each class
    pointsCls1 = X(resp1 > 0.5,:);
    pointsCls2 = X(resp2 > 0.5,:);
    
    % draw graph
    hold on
    scatter(x, y, 10, colors);
    hold off
    
    % draw title
    title(sprintf('iteration %d', iteration));
end

%% Contour Level
function level = contourLevel(X, mu, sigma)
    N = size(X, 1);
    level = zeros(N,2);
    for i = 1:N
        x = X(i,:);
        level(i, :) = mvnpdf(x, mu, sigma);
    end
end

