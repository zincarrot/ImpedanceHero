function plotyreal_all(data)
f=data(1,:);
c=jet(size(data,1));
for k=2:size(data,1)
    y=1./data(k,:);
semilogx(f,real(y),'Color',c(k,:));
hold on;
end
xlabel("f/Hz");
ylabel("\theta/\circ");
set(gca,'Ydir','reverse');
end