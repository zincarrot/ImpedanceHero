function two_component_fit_all2(data,x0)
f=data(1,:);
x_ini=x0;
result_para=zeros([size(data,1)-1,8]);
result_r2=zeros([size(data,1)-1,1]);
for i=2:size(data,1)
    fprintf('fitting spectrum #%d',i-1);
    [x_ini,~,r2]=two_component_fit2(f,data(i,:),x_ini);
    result_para(i-1,:)=x_ini;
    result_r2(i-1)=r2;
    assignin('base','result2_para_',result_para);
    assignin('base','result2_r2',result_r2);
end