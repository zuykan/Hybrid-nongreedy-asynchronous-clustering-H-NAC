function [J_NAC_cls_vec, Nk_vec] = function_HNACquantzError_vec(X_tr, ymax, label, alpha)
nrClus = length(label);
quantzn_Error_vec_squared = zeros(1, nrClus);
C_kGADIA_update = [];
Nk_vec = [];
J_NAC = 0;
J_NAC_cls_vec = [];
for kk = 1:nrClus
    %Calculate the sum of intra-cluster distances:
    tmp_inds = [];
    tmp_inds = find( ymax == label(kk) );
    C_kGADIA_update(:, kk) = mean( X_tr(:, tmp_inds), 2 );
    Nk = length(tmp_inds);
    for s = 1 : Nk
        quantzn_Error_vec_squared(kk) = quantzn_Error_vec_squared(kk) + norm( X_tr(:,tmp_inds(s))-C_kGADIA_update(:, kk) )^2;        
    end
    Nk_vec = [Nk_vec  Nk];
    J_NAC = J_NAC + quantzn_Error_vec_squared(kk)*(alpha*(2*Nk-1) + 1);
    J_NAC_cls_vec = [ J_NAC_cls_vec  quantzn_Error_vec_squared(kk)*(alpha*(2*Nk-1) + 1) ];
end

