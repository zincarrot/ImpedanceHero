function plotmag_all(data)
f=data(1,:);
for k=2:size(data,1)
    t=abs(data(k,:));
loglog(f,t);
hold on;
end
xlabel("f/Hz");
ylabel("|Z|/\Omega");
end