function RCRC_slider()

R1 = 10;
R2 = 10000;
C1= 1e-10;
C2= 1e-6;
a=-1;
b=-1;
c=1;

f_exp=linspace(0,10,100);
f=10.^f_exp;

%z = RCRC(R1,C1,R2,C2,a,b,c,f); 

hplot=plotecplx(R1+1000j,f,1);
R1 = 10:1000;
h1 = uicontrol('style','slider','units','pixel','position',[20 20 300 20]);
addlistener(h1,'ContinuousValueChange',@(hObject, event) makeplot(hObject, event,R1,hplot));
end

function makeplot(hObject,event,x,hplot)
n = get(hObject,'Value');
set(hplot,'ydata',x.^n);
drawnow;
end
