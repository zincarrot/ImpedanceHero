function [r,n,Q]=guessCPE(z1,f,lfb,ufb)

%% frequency filter
if nargin<3
    lfb=0;
    ufb=inf;
end
data_idx=(f>=lfb)&(f<=ufb);
f_select=fdata(data_idx);
ndata=size(z1);

%% initial guess (CPE)
[hif, hif_ind]=max(f_select);
[lof, lof_ind]=min(f_select);
hi_imp=data(2,hif_ind);
lo_imp=data(2,lof_ind);
r=real(hi_imp)+imag(hi_imp)*real(hi_imp-lo_imp)/imag(hi_imp-lo_imp);
n=log(imag(hi_imp)/imag(lo_imp))/log(lof/hif);
Q=imag(1/((2j*pi*hif)^n_init))/imag(hi_imp);