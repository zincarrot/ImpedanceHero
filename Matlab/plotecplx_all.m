function plotecplx_all(data, colorcode)

if nargin == 1
    f=data(1,:);
    c=jet(size(data,1));
    for k=2:size(data,1)
        t=1./(1i*2*pi*f.*(data(k,:)));
        plot(real(t),imag(t),'Color',c(k,:));
        
        hold on;
        axis equal
    end

elseif colorcode == "default"
    f=data(1,:);
    for k=2:size(data,1)
        t=1./(1i*2*pi*f.*(data(k,:)));
        plot(real(t),imag(t));
        
        hold on;
        axis equal
    end
end

xlabel("C'/F");
ylabel("C''/F");
set(gca,'Ydir','reverse');
end