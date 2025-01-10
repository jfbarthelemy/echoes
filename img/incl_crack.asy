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

pen THICK = linewidth(2.0 * linewidth());
pen THIN = linewidth(0.7 * linewidth());
pen VTHICK = linewidth(3.0 * linewidth());

size(200,300,keepAspect=true);

real a=3., b=1.5, c=0.;

currentprojection=orthographic(10,3,3);
currentlight.background=black+opacity(0.0);

// currentlight=(10,10,5);
triple f(pair t) {return (a*sin(t.x)*cos(t.y),b*sin(t.x)*sin(t.y),c*cos(t.x));}
surface s=surface(f,(0,0),(pi,2pi),8,16,Spline);
// s.colors(palette(s.map(zpart),Rainbow()+opacity(0.5)));
draw(s,heavyblue+opacity(0.5),render(merge=true));

real x(real t) {return a*cos(pi*t);}
real y(real t) {return b*sin(pi*t);}
real z(real t) {return 0;}
path3 p=graph(x,y,z,0,2,operator ..);
draw(p,heavyred+dashed);

draw((0,0,0)--(a,0,0),heavyred+dashed);
draw((0,0,0)--(0,b,0),heavyred+dashed);
draw((0,0,0)--(0,0,1),heavyblue+VTHICK,Arrow3);
