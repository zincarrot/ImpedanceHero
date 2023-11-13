function plotmag_all(data, colorcode)
f=data(1,:);
c=jet(size(data,1));
for k=2:size(data,1)
    t=abs(data(k,:));
loglog(f,t,'Color',c(k,:));
hold on;
end

if nargin == 1
    f=data(1,:);
    c=jet(size(data,1));
    for k=2:size(data,1)
        t=abs(data(k,:));
        loglog(f,t,'Color',c(k,:));
        
        hold on;
    end

elseif colorcode == "default" || colorcode == "d"
    f=data(1,:);
    for k=2:size(data,1)
        t=abs(data(k,:));
        loglog(f,t);
        
        hold on;
    end
end
xlabel("f/Hz");
ylabel("|Z|/\Omega");
end