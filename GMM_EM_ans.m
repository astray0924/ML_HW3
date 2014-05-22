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

%% Select Assumption for Covariance Matrix
COV_ASSUMPT = 0;
% 0 : for full covariance matrix (default)
% 1 : for independent isotropic, sigma * eye(D).
% 2 : for independent diagonal, diag([...]);

%% Define procedure for drawing graph
color = [1 0 0; 0 0 1];

    function contourPlot(param, color)
        SigHalf = sqrtm(param.Sigma);
        
        for i=1:101
            deg=i/100;
            v = [cos(2 * pi * deg); sin(2 * pi * deg)];
            z(:,i) = param.mu + SigHalf * v;
        end

        plot(z(1,:),z(2,:),'r','color', color, 'linewidth', 2)
    end

    function graph(isMaximized)
        hFig = figure;
        set(hFig, 'Position', [100 100 500 500]);
        axis([-3, 3, -3, 3]);
        hold on;
        if isMaximized
            title(strcat('Iteration ', num2str(step)));
        else
            title(strcat('After E-step, Iteration ', num2str(step)));
        end
        %for i=1:N
        if resp(i, 1) + resp(i, 2) == 0
            scatter(X(:, 1), X(:, 2), 10, [0, 0.5, 0], 'fill');
        else
            scatter(X(:, 1), X(:, 2), 10, resp * color, 'fill');
        end
        for k=1:K
            contourPlot(Theta(k), color(k, :, :));
        end
        hold off;
    end

%% Initialization
for i=1:K
    Theta(i).pi = 1/K;
    Theta(i).Sigma = eye(D);
end
Theta(1).mu = [-1; 1];
Theta(2).mu = [1; -1];

epsilon = 1e-4;
step = 0;
resp = zeros(N, K);

%% Compute E-M.
while 1
    if step < 6 || step == 16
        graph(true);
    end
    
    if (step > 1 && abs(Q(step) - Q(step - 1)) < epsilon)
        break;
    end
    step = step + 1;
    
    %% Expectation step
    for i=1:N
        s = 0;
        for k=1:K
            resp(i, k) = Theta(k).pi * mvnpdf(X(i, :)', Theta(k).mu, Theta(k).Sigma);
            s = s + resp(i, k);
        end
        resp(i, :) = resp(i, :) / s;
    end
    
    if step == 1
        graph(false);
    end
    
    graph(false);
    
    %% Maximization step
    sumResp = sum(resp);
    for k=1:K
        Theta(k).pi = sumResp(k) / N;
        Theta(k).mu = zeros(D, 1);
        
        if COV_ASSUMPT == 0
            % LINE FOR DEFAULT
            Theta(k).Sigma = zeros(D, D);
        elseif COV_ASSUMPT == 1
            % LINE FOR ISOTROPIC
            sigmaSum = 0;
        else
            % LINE FOR DIAGONAL
            Theta(k).sigma = zeros(D, 1);
        end
        
        for i=1:N
            x = X(i, :)';
            
            Theta(k).mu = Theta(k).mu + resp(i, k) * x;
            
            if COV_ASSUMPT == 0
                % LINE FOR DEFAULT
                Theta(k).Sigma = Theta(k).Sigma + resp(i, k) * x * x';
            elseif COV_ASSUMPT == 1
                % LINE FOR ISOTROPIC
                sigmaSum = sigmaSum + resp(i, k) * sum( x .^ 2 );
            else
                % LINE FOR DIAGONAL
                Theta(k).sigma = Theta(k).sigma + resp(i, k) * ( x .^ 2 );
            end
        end
        Theta(k).mu = Theta(k).mu / sumResp(k);
        
        if COV_ASSUMPT == 0
            % LINE FOR DEFAULT
            Theta(k).Sigma = Theta(k).Sigma / sumResp(k) - Theta(k).mu * Theta(k).mu.';
        elseif COV_ASSUMPT == 1
            % GROUP FOR ISOTROPIC
            sigmaSum = ( sigmaSum / sumResp(k) - sum( Theta(k).mu .^ 2 ) ) / D;
            Theta(k).Sigma = eye(D) * sigmaSum;
        else
            % GROUP FOR DIAGONAL
            Theta(k).sigma = Theta(k).sigma / sumResp(k) - ( Theta(k).mu .^ 2 );
            Theta(k).Sigma = diag( Theta(k).sigma );
        end
    end
    Theta(2).Sigma

    %% Convergence check
    Q(step) = 0;
    for i=1:N
        for k=1:K
            
            % If Sigma is not symmetric, then there is an arithmetic error.
            [~,err]=cholcov(Theta(k).Sigma);
            if isnan(err)
                Theta(k).Sigma = (Theta(k).Sigma + Theta(k).Sigma.') / 2;
            end
            
            Q(step) = Q(step) + resp(i, k) * (log(Theta(k).pi) + log(mvnpdf(X(i, :)', Theta(k).mu, Theta(k).Sigma)));
        end
    end
end
% 
% graph(true);
% 
% figure;
% hold on;
% plot(1:step, Q, '-', 'color', [0,0,0]);
% hold off;
end

