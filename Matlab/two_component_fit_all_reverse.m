function two_component_fit_all_reverse(data,x0)
f=data(1,:);
x_ini=x0;
result_para=zeros([size(data,1)-1,8]);
result_r2=zeros([size(data,1)-1,1]);
for i=size(data,1):-1:2
    fprintf('fitting spectrum #%d',i-1);
    [x_ini,~,r2]=two_component_fit(f,data(i,:),x_ini);
    result_para(i-1,:)=x_ini;
    result_r2(i-1)=r2;
    assignin('base','result_para_reverse_x176',result_para);
    assignin('base','result_r2_reverse_x176',result_r2);
end