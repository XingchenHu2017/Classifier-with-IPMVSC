function [UU,V,A,Ao,W,Z,alpha,iter,obj] = algo_chd(X,lambda,numanchor)
% Multi-view projected clustering for high dimensional data
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

m = numanchor;
numview = length(X);
numsample = size(X{1},1);

W = cell(numview,1);            % dt * dt
A = cell(numview,1);            % dt * m
Z = zeros(m,numsample);         % m  * N

for t = 1:numview
   dt = size(X{t},2); 
   W{t} = zeros(dt,dt);
   A{t} = zeros(dt,m);         % d  * m
%    X{t} = X{t}';
   [X{t} PS{t}] = mapstd(X{t}',0,1); % turn into d*n
end
Z(:,1:m) = eye(m);


alpha = ones(1,numview)/numview;
opt.disp = 0;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;
    
    %% optimize W_i

    parfor iv=1:numview
        AZ{iv} = A{iv}*Z;
        C{iv} = X{iv}*AZ{iv}';      
        [U,~,V] = svd(C{iv},'econ');
        W{iv} = U*V';
    end

    %% optimize A
    for ia = 1:numview
        al2 = alpha(ia)^2;
        D{ia} = al2 * W{ia}' * X{ia} * Z';
        [U,~,V] = svd(D{ia},'econ');
        A{ia} = U*V';
        Ao{ia} = mapstd('reverse',A{ia},PS{ia});
    end
    
    %% optimize Z
    sumAlpha = sum(alpha.^2);
    % % QP
% Sbar=[];
% H = 2*sumAlpha*A'*A+2*lambda*eye(m);
H = 2*sumAlpha*eye(m)+2*lambda*eye(m);
H = (H+H')/2;
% [r,q] = chol(H);

options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm 默认为 interior-point-convex
parfor ji=1:numsample
    ff=0;
    for j=1:numview
%         C{j} = W{j} * A{j};
%         ff = ff - 2*X{j}(:,ji)'*C{j};
        C = W{j} * A{j};
        ff = ff - 2*X{j}(:,ji)'*C;
    end
    Z(:,ji) = quadprog(H,ff',[],[],ones(1,m),1,zeros(m,1),ones(m,1),[],options);
end

    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm( X{iv} - W{iv} * A{iv} * Z,'fro');
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A{iv} * Z,'fro')^2;
    end
    term2 = lambda * norm(Z,'fro')^2;
    obj(iter) = term1+ term2;
    
    
    if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(Z','econ');
        flag = 0;
    end
end
         
         
    
