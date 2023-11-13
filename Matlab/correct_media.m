function z_modified=correct_media(data,media)
z_modified=data;
z_modified(2:end,:)=data(2:end,:)-media(3,:);
end