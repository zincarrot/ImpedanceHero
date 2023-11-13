function [data_mean,data_std] = getz(filename)
raw=readmatrix(filename);
raw_reshape=reshape(raw,[size(raw,1),5,size(raw,2)/5]);
real=reshape(raw_reshape(:,3,:),[size(raw,1),size(raw,2)/5]);
imag=reshape(raw_reshape(:,4,:),[size(raw,1),size(raw,2)/5]);
data=real+1j*imag;
data(data==0)=nan;
data(data>1e30)=nan;
data_mean=mean(data,1,"omitnan");
if nargout>1
data_std=std(data,0,1,"omitnan");
end
end
