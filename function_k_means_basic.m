function [ ymax, C_kmeans_update, quantzError_kmeans_update_2 ] = function_k_means_basic( X_tr, ...    C_init, maxStepNumber, init_y_cls_VQ, isplot_simuCampaign, figNr )quantzError_kmeans_update = -999;quantzError_kmeans_update_2 = -999;N = size(X_tr, 2);ymax = init_y_cls_VQ;nrClus = size(C_init,2);label_vec = [1:nrClus];if (isplot_simuCampaign)    figure(figNr), plot( X_tr(1,:), X_tr(2,:), 'k.', 'MarkerSize', 7 ), hold on    figure(figNr), plot( C_init(1,:), C_init(2,:), 'bd', 'MarkerSize', 8 ), hold on    legend('data points', 'centroids of k-means')endC_kmeans_update = C_init;n_epoch = 1;continueIter = true;while (n_epoch < maxStepNumber) && (continueIter)    ymax_prev = ymax;    C_kmeans_update_prev = C_kmeans_update;        %%%%%%%%% Step 1: Expectation step    E_cev_vq = 0;    for s = 1:N        uz_vq = [];        for k = 1:nrClus            uz_vq(k) = norm( X_tr(:,s) - C_kmeans_update(:,k) )^2;        end        [ ali_vq z_vq ] = min( uz_vq );        ymax(s) = z_vq;        E_cev_vq = E_cev_vq + norm( X_tr(:,s)-C_kmeans_update(:,z_vq) )^2;    end %s=1:L_tr    %quantzError_kmeans_update_2 = [quantzError_kmeans_update_2  E_cev_vq];        %%%%%%%%% enf of step 1    %After ALL data points cluster indices have been updated in step 1, we update the centroids in step 2.    %Step 2: Maximization step    for jkl = 1:nrClus        tmpinxvctr = [];        tmpinxvctr = find(ymax == jkl);        if isempty(tmpinxvctr) %if any cluster is EMPTY, then we give a sample to this cluster so that it is not empty.            tmpindx = jkl*N/nrClus;            ymax(tmpindx) = jkl;            C_kmeans_update(:, jkl) = X_tr(:, tmpindx);        else            C_kmeans_update(:, jkl) = mean( X_tr(:, tmpinxvctr), 2 );        end            end    %epoch ended!    n_epoch = n_epoch +1;    %if ( norm(ymax_prev - ymax) == 0 )    %if ( norm(C_kmeans_update_prev - C_kmeans_update) == 0 )    if ( norm( max(C_kmeans_update_prev - C_kmeans_update) ) == 0 )        continueIter = ~true;            end    end