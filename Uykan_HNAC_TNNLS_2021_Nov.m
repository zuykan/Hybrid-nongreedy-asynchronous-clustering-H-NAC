%Z. Uykan, "Fusion of Centroid-Based Clustering With Graph Clustering:
%An Expectation-Maximization-Based Hybrid Clustering," in IEEE Transactions on Neural Networks
%and Learning Systems, doi: 10.1109/TNNLS.2021.3121224.

close all
clear all

global isplot_simuCampaign
isplot_simuCampaign = 1; %1; %0; %1;

dataType_nr = 3;
%3 = MNIST

if (dataType_nr == 3)
    %MNIST
    %load Uykan_HNAC_data_MNIST_100.mat; k_clust_vec = [ 10 ]; NrDocsPerCluster = 100;
    
    %CLATECT-101
    load Uykan_HNAC_data_Caltech101_7.mat; k_clust_vec = [ 7 ]; NrDocsPerCluster = 34;
end

%to remove the mean value from the data
meanDoc = [];
meanDoc = mean(X_tr_ori, 2); %Centroid: Average of all documents in the same cluster
for mmn = 1:size(X_tr_ori,2)
    X_tr_ori(:,mmn) = X_tr_ori(:,mmn) - meanDoc;
end

RESULTS_ACC_NMI_Pur_NAC2 = [];
k_clust = k_clust_vec(1);
alpha_vec = [0 1 2 3 4 5 6 10]/NrDocsPerCluster;

%function_HNAC.m
L_tr = size(X_tr_ori, 2); % L_tr : number of samples for training
rng(12345)

k_clust = 7; %number of clusters
tmp_index = randperm(L_tr); %choosing the initial centroids from the data vectors randomly.
C_init = X_tr_ori( : , tmp_index(1:k_clust) );

k_clust = size(C_init, 2);
init_y_cls_VQ = [];
for s=1:L_tr
    uz_vq = [];
    for k = 1:k_clust
        uz_vq(k) = norm(X_tr_ori(:,s)-C_init(:,k))^2;
    end
    [veli_vq z_vq] = min(uz_vq);
    init_y_cls_VQ(s) = z_vq;
end

%H-NAC
figNr = 400;
maxStepNumber = 100;

%%%%%% %for GRAPH CLUSTERING PART only:  true: Euclidean distance, and false:Cosine distance
N = L_tr;
isEuclideanDist = true; %~true;
W_ori = [];
if (isEuclideanDist)
    %squared Euclidean distance
    for jj=1:N
        for kk=jj:N
            W_ori(jj,kk) = norm( X_tr_ori(:,jj) - X_tr_ori(:,kk) )^2;
        end
    end
    W_ori = W_ori + W_ori'; %diagonal elements are always zero!
    
else
    %Cosine distance
    cntr = mean( X_tr_ori, 2 );
    for jj=1:N
        P(:,jj) = ( X_tr_ori(:,jj) - cntr );
    end
    
    W_cos = [];
    for nn=1:N
        for mm=1:N
            %Cosine similarity
            cossim_n_m = P(:,nn)' * P(:,mm) / ( norm(P(:,nn)) * norm(P(:,mm)) );
            
            %Cosine distance
            W_ori(nn,mm) = 1 - cossim_n_m; %cosine distance
        end
    end
    
end


for jjkk=1:length(alpha_vec)
    y_indx_vec_NAC2_opt = [];
    alpha = alpha_vec(jjkk);
    
    %function_HNAC 
    strTitle = strcat('\alpha =', num2str(alpha)); %'title';
    
    isInstant_Ck_update = false; %true; %false;
    [ y_indx_vec_NAC2,  C_NAC2, quantzError_NAC2] = function_HNAC_2021TNNLS( X_tr_ori, ...
        C_init, maxStepNumber, init_y_cls_VQ, isplot_simuCampaign, figNr, alpha, isInstant_Ck_update, strTitle, W_ori );
    [res_ACC_NMI_Pur_NAC2] = ClusteringMeasure_Kang( y_indx_vec_NAC2, y_indx_desired_vec );
    
    
    jjkk
    RESULTS_ACC_NMI_Pur_NAC2(jjkk, :) = [ res_ACC_NMI_Pur_NAC2 ];
    
end
 
mrkrSz = 5;
lWdth = 3;
fntSzL = 14;
%batch mode
nrOfAlphas = length(alpha_vec);
figure, bar( RESULTS_ACC_NMI_Pur_NAC2(1:nrOfAlphas, :) )
set(gca, 'FontName', 'Arial', 'FontWeight', 'bold', 'FontAngle', 'normal', 'FontSize', fntSzL);
title('Clustering Performance with respect to \alpha  - batch mode')
legend('ACC', 'NMI', 'Purity')
xlabel('\alpha', 'FontName', 'Helvetica', 'FontSize', 20, 'FontWeight', 'bold' )
xticklabels({alpha_vec(1), alpha_vec(2),alpha_vec(3),alpha_vec(4),alpha_vec(5),...
    alpha_vec(6),alpha_vec(7),alpha_vec(8)})
ylabel( 'ACC, NMI, Purity', 'FontName', 'Helvetica', 'FontSize', fntSzL, 'FontWeight', 'bold' )
grid,
