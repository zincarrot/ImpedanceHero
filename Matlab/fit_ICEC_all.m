function fit_ICEC_all(data,x0)
f=data(1,:);
x_ini=x0;
result_para=zeros([size(data,1)-1,5]);
result_r2=zeros([size(data,1)-1,1]);
for i=2:size(data,1)
    fprintf('fitting spectrum #%d',i-1);
    [x_ini,~,r2]=fit_ICEC(f,data(i,:),x0);
    result_para(i-1,:)=x_ini;
    result_r2(i-1)=r2;
    assignin('base','result_para_x_1',result_para);
    assignin('base','result_r2_x_1',result_r2);
end