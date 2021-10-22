function [fmse,fgrad]=model_mse_build(model,handle,dim,name)
    data=sym('d',[1 dim]);
    mse=mean((log(model)-log(data)).*conj(log(model)-log(data)));
    grad=real(gradient(mse, handle));
    disp('constructing mse function...')
    fmse=matlabFunction(mse,'File',strcat('mse_',name));
    disp('constructing gradient function...')
    fgrad=matlabFunction(grad,'File',strcat('grad_',name));
end