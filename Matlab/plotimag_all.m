function plotimag_all(data)
c=jet(size(data,1));
f=data(1,:);
for k=2:size(data,1)
semilogx(f,imag(data(k,:)),'Color',c(k,:));
hold on;
end
xlabel("f/Hz");
ylabel("Z''/\Omega");
set(gca,'Ydir','reverse');
end