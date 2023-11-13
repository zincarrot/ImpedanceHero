function plotmreal_all(data)
f=data(1,:);
c=jet(size(data,1));
for k=2:size(data,1)
    t=1i*2*pi*f.*(data(k,:));
semilogx(f,real(t),'Color',c(k,:));
hold on;
end
xlabel("f/Hz");
ylabel("\theta/\circ");
end