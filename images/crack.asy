// settings.outformat = "html";
// settings.render = -1;

// settings.outformat ="svg";
// settings.render = 0;
// Hint: render svg format and then convert to png by "magick -density 300 file.svg file.png"

settings.outformat = "pdf";
settings.render = 8;
settings.prc = true;

import graph3;
import palette;
import three;

size(200,300,keepAspect=true);

real a=3., b=1.5, c=0.1;

currentprojection=orthographic(10,10,5);
// currentlight=(10,10,5);
triple f(pair t) {return (a*sin(t.x)*cos(t.y),b*sin(t.x)*sin(t.y),c*cos(t.x));}

surface s=surface(f,(0,0),(pi,2pi),8,16,Spline);
// s.colors(palette(s.map(zpart),Rainbow()+opacity(0.5)));

real x(real t) {return a*cos(2pi*t);}
real y(real t) {return 0;}
real z(real t) {return c*sin(2pi*t);}
path3 p=graph(x,y,z,-1,1,operator ..);
draw(p,red+dotted);

real x(real t) {return 0;}
real y(real t) {return b*cos(2pi*t);}
real z(real t) {return c*sin(2pi*t);}
path3 p=graph(x,y,z,-1,1,operator ..);
draw(p,red+dotted);

draw(s,lightgreen+opacity(0.2),render(merge=true));

draw("\scriptsize $\underline{\ell}$",(0,0,0)--(1,0,0),blue,Arrow3);
draw("\scriptsize $\underline{m}$",(0,0,0)--(0,1,0),NE,blue,Arrow3);
draw("\scriptsize $\underline{n}$",(0,0,0)--(0,0,1),blue,Arrow3);

draw((0,0,0)--(a,0,0),red+dashed);
draw((0,0,0)--(0,b,0),red+dashed);
draw((0,0,0)--(0,0,c),red+dashed);

draw(Label("\scriptsize $a$"),(a,0,0),N,black);
draw(Label("\scriptsize $b=\eta\,a$"),(0,b,0),E,black);
draw(Label("\scriptsize $c=\omega\,a$"),(0,0,c),NE,black);
