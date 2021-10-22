function plotang(f,z)
semilogx(f,rad2deg(angle(z)));
ylabel("\theta/\circ");
xlabel("f/Hz");
set(gca,'Ydir','reverse');
end