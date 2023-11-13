function z_modified=normalize_CPE(data,CPE_param)
z_modified=data;
for i=1:size(data,1)-1
    z_modified(i+1,:)=data(i+1,:).*CPE_param(i,3);
end