function ploteang_all(data)
f=data(1,:);
for k=2:size(data,1)
    t=1./(1i*2*pi*f.*(data(k,:)));
semilogx(f,rad2deg(angle(t)));
hold on;
end
xlabel("f/Hz");
ylabel("\theta/\circ");
end