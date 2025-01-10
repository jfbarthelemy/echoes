settings.outformat = "html";
settings.render = -1;

// settings.outformat ="svg";
// settings.render = 0;
// // Hint: render svg format and then convert to png by "magick -density 300 file.svg file.png"

// settings.outformat = "pdf";
// settings.render = 8;
// settings.prc = true;

import graph3;
import palette;
import three;

size(200,300,keepAspect=true);

real L=3., b=1.5, c=0.8;

currentprojection=orthographic(10,10,5);
currentlight.background=black+opacity(0.0);

// currentlight=(10,10,5);
triple f(pair t) {return (t.x,b*cos(2pi*t.y),c*sin(2pi*t.y));}
surface s=surface(f,(-L,0),(L,1),8,16,Spline);
// s.colors(palette(s.map(zpart),Rainbow()+opacity(0.5)));
draw(s,heavyblue+opacity(0.5),render(merge=true));

real x(real t) {return -L;}
real y(real t) {return b*cos(pi*t);}
real z(real t) {return c*sin(pi*t);}

path3 p=graph(x,y,z,0,2,operator ..);
draw(p,heavyred+dashed);

real x(real t) {return L;}
path3 p=graph(x,y,z,0,2,operator ..);
draw(p,heavyred+dashed);

// draw("\scriptsize $\vec{l}$",(0,0,0)--(1,0,0),S,royalblue,Arrow3);
// draw("\scriptsize $\vec{m}$",(0,0,0)--(0,1,0),S,royalblue,Arrow3);
// draw("\scriptsize $\vec{n}$",(0,0,0)--(0,0,1),NE,royalblue,Arrow3);

// draw((-L,0,0)--(L,0,0),heavyred+dotted);
// draw((-L,0,0)--(-L,b,0),heavyred+dotted);
// draw((-L,0,0)--(-L,0,c),heavyred+dotted);

// label(Label("\scriptsize $L\to\infty$"),(0.65L,0,0),black);
// label(Label("\scriptsize $b$"),(-L,b,0),E,black);
// label(Label("\scriptsize $c=\rho\,b$"),(-L,0,c),NE,black);
