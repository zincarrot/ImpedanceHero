function plotang_all(data)
f=data(1,:);
c=jet(size(data,1));
for k=2:size(data,1)
    t=rad2deg(angle(data(k,:)));
semilogx(f,t,'Color',c(k,:));
hold on;
end
xlabel("f/Hz");
ylabel("\theta/\circ");
set(gca,'Ydir','reverse');
end