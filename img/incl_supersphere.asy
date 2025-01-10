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

currentprojection=orthographic(10,10,5);
currentlight.background=black+opacity(0.0);

typedef triple surffunc(pair);

surffunc gensupsphfunc(real p, real a, real b, real c) {
return new triple(pair t) {
    real ux = sin(t.x)*cos(t.y) ;
    real uy = sin(t.x)*sin(t.y) ;
    real uz = cos(t.x) ;
    return (a*sgn(ux)*abs(ux)**(1./p),b*sgn(uy)*abs(uy)**(1./p),c*sgn(uz)*abs(uz)**(1./p));
    };
}

surffunc gensupspherofunc(real p, real a, real c) {
return new triple(pair t) {
    real ur = sin(t.x);
    real uz = cos(t.x) ;
    return (a*sgn(ur)*abs(ur)**(1./p)*cos(t.y),a*sgn(ur)*abs(ur)**(1./p)*sin(t.y),c*sgn(uz)*abs(uz)**(1./p));
    };
}

surface s=surface(gensupsphfunc(0.3,1,1,1),(0,0),(pi,2pi),32,64,Spline);
// surface s=surface(gensupspherofunc(0.3,1,1),(0,0),(pi,2pi),32,64,Spline);
draw(s,heavyblue+opacity(0.5),render(merge=true));

// real x(real t) {return a*cos(pi*t);}
// real y(real t) {return b*sin(pi*t);}
// real z(real t) {return 0;}
// path3 p=graph(x,y,z,0,0.5,operator ..);
// draw(p,heavyred+dashed);

// real x(real t) {return a*cos(pi*t);}
// real y(real t) {return 0;}
// real z(real t) {return c*sin(pi*t);}
// path3 p=graph(x,y,z,0,0.5,operator ..);
// draw(p,heavyred+dashed);

// real x(real t) {return 0;}
// real y(real t) {return b*cos(pi*t);}
// real z(real t) {return c*sin(pi*t);}
// path3 p=graph(x,y,z,0,0.5,operator ..);
// draw(p,heavyred+dashed);

// // draw((0,0,0)--(0.3,0,0),royalblue,Arrow3);
// // draw((0,0,0)--(0,0.3,0),royalblue,Arrow3);
// // draw((0,0,0)--(0,0,0.3),royalblue,Arrow3);

// draw((0,0,0)--(a,0,0),heavyred+dashed);
// draw((0,0,0)--(0,b,0),heavyred+dashed);
// draw((0,0,0)--(0,0,c),heavyred+dashed);

// label(Label("\scriptsize $a$"),(a,0,0),N,black);
// label(Label("\scriptsize $b=\eta\,a$"),(0,b,0),E,black);
// label(Label("\scriptsize $c=\omega\,a$"),(0,0,c),NE,black);
