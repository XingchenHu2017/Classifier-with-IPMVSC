%-------------------------------------------------------------------------------------------------------------------------------------
% Contributed by Xingchen Hu
% Ref:
% [1] Multi-view Fuzzy Classification with Subspace Clustering and
%     Information Granules. (TKDE, 2022).
% [2] Granular fuzzy rule-based models: A study in a comprehensive
%     evaluation and construction of fuzzy models (TFS, 2017)
%-------------------------------------------------------------------------------------------------------------------------------------
clear
clc
data_name = 'HW';
load(strcat('.\',data_name,'\flag_group.mat'))
load(strcat('.\',data_name,'\data.mat'))
addpath(genpath('./'));

nclass = max(Y);
lambda = 0;
nview = length(X);

c_num = 1;
for c = 12%3:4:27
    q_num = 1;
    for  q = 1.1%1.1:0.3:2.9
        fprintf('%s c = %0.2f m = %0.2f\n',data_name,c,q)
        for fold = 1:1
            
            % load training and test data sets
            Train_flag = Trainflag{fold};
            Test_flag = Testflag{fold};
            
            trn_temp = Y(Train_flag,:);
            trn_targets = ones(size(trn_temp,1),nclass)*-1;
            for i = 1:size(trn_temp,1)
                trn_targets(i,trn_temp(i)) = 1;
            end
            for t = 1:nview
                temp = (X{t});    % transport dimensions of input data if needed
                trn_inputs{t} = temp(Train_flag,:);
                trn_data{t} = ([temp(Train_flag,:) trn_targets]);
            end
            
            for t = 1:nview
                temp = (X{t});     % transport dimensions of input data if needed
                tst_inputs{t} = temp(Test_flag,:);
            end
            
            tst_temp = Y(Test_flag,:);
            tst_targets = ones(size(tst_inputs{1},1),nclass)*-1;
            for i = 1:size(tst_inputs{1},1)
                tst_targets(i,tst_temp(i)) = 1;
            end
            
            
            %% fuzzy clustering
            Alpha = ones(nview,1)/nview;
            
            anchor = 1*c;   %[k 2*k 3*k]
            d = c;
            
            profile on
            [U,V,A,VW,W,Z,alpha_n,iter,obj] = algo_chd(trn_data,lambda,anchor); % X,Y,lambda,d,numanchor
            
            res = myNMIACCwithmean(U,trn_temp,c);
            ACC_folds(fold) = res(1);
            NMI_folds(fold) = res(2);
            Purity_folds(fold) = res(3);
            
            %% fuzzy modeling
            for t = 1:nview
                Vi{t} = (VW{t}(1:end-nclass,:))';
                Wi{t} = (VW{t}(end-nclass+1:end,:))';
            end
            
            trn_zz_mv = [];
            trn_h_mv = [];
            for t = 1:nview
                D=size(trn_inputs{t},2);
                % training data output
                Mem =partition_matrix(Vi{t},trn_inputs{t},q);
                row = size(Vi{t},1);
                num = size(trn_inputs{t},1);
                for i = 1:row
                    for k = 1:num
                        trn_zz(i*(D)-(D-1):i*(D),k) = (Mem(i,k)*(trn_inputs{t}(k,:)-Vi{t}(i,:)))';
                    end
                end
                for l = 1:nclass
                    for k = 1:num
                        for i = 1:row
                            trn_h_temp(k,i) = Mem(i,k)*Wi{t}(i,l);
                        end
                    end
                    trn_h(:,l) = sum(trn_h_temp,2);
                end
                trn_zz = trn_zz';
                aa{t} =  (trn_zz'*trn_zz+eye(size(trn_zz,2)))\trn_zz'*(trn_targets-trn_h);
                %  aa{t} = pinv(trn_zz)*(trn_targets-trn_h);
                trn_output_view{t} = (trn_h+trn_zz*aa{t});
                
                clear trn_zz trn_h_temp trn_h
            end
            
            trn_outputs = zeros(size(trn_temp,1),nclass);
            for t = 1:nview
                trn_outputs = trn_outputs+trn_output_view{t};
            end
            trn_output = trn_outputs/nview;
            [~,yPredTrn] = max(trn_output,[],2);
            AccTrain(fold)=mean(yPredTrn==Y(Train_flag));
            
            % testing data output
            for t = 1:nview
                D=size(tst_inputs{t},2);
                Mem =partition_matrix(Vi{t},tst_inputs{t},q);
                num = size(tst_inputs{t},1);
                for i = 1:row
                    for k = 1:num
                        tst_zz(i*(D)-(D-1):i*(D),k) = (Mem(i,k)*(tst_inputs{t}(k,:)-Vi{t}(i,:)))';
                    end
                end
                for l = 1:nclass
                    for k = 1:num
                        for i = 1:row
                            tst_h_temp(k,i) = Mem(i,k)*Wi{t}(i,l);
                        end
                    end
                    tst_h(:,l) = sum(tst_h_temp,2);
                end
                tst_zz = tst_zz';
                tst_output_view{t} = (tst_h+tst_zz*aa{t});
                
                clear tst_zz tst_h_temp tst_h
            end
            tst_outputs = zeros(size(tst_temp,1),nclass);
            for t = 1:nview
                tst_outputs = tst_outputs+tst_output_view{t};
            end
            trn_output = tst_outputs/nview;
            [~,yPredTst] = max(trn_output,[],2);
            AccTest(fold)=mean(yPredTst==Y(Test_flag,end));
            
        end
        ACCTest_all{c_num,q_num} = AccTest;
        ACC_mean(c_num,q_num) = mean(AccTest);
        q_num = q_num+1;
    end
    c_num = c_num+1;
end