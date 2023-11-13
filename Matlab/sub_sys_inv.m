function data_sub=sub_sys_inv(data,sys)
data_inv=data(2:end,:);
sys_inv=sys(2:end,:);
data_sub=data;
data_sub(2:end,:)=1./(1./data_inv-1./sys_inv);
data_sub(1,:)=data(1,:);
end