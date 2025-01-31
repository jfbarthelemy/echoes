// settings.outformat = "html";
// settings.render = -1;

settings.outformat ="svg";
// settings.render = 0;
// Hint: render svg format and then convert to png by "magick -density 300 file.svg file.png"

// settings.outformat = "pdf";
// settings.render = 8;
// settings.prc = true;

import graph3;
import palette;
import three;

srand(1949);

size(200,300,keepAspect=true);

real R=30., a=3., b=1.5, c=0.1;

currentprojection=orthographic(10,10,5);
// currentlight=(10,10,5);

typedef triple surffunc(pair);

surffunc genellfunc(triple r, triple c, real omega, triple n) {
return new triple(pair t) {return c+rotate(omega,n)*(r.x*sin(t.x)*cos(t.y),r.y*sin(t.x)*sin(t.y),r.z*cos(t.x));};
}

surffunc gensupsphfunc(real p, triple r, triple c, real omega, triple n) {
return new triple(pair t) {
    real ux = sin(t.x)*cos(t.y) ;
    real uy = sin(t.x)*sin(t.y) ;
    real uz = cos(t.x) ;
    return c+rotate(omega,n)*(r.x*sgn(ux)*abs(ux)**(1./p),r.y*sgn(uy)*abs(uy)**(1./p),r.z*sgn(uz)*abs(uz)**(1./p));
    };
}

triple genunit() {
    real z = 2*unitrand()-1 ;
    real r = sqrt(1-z*z) ;
    real phi = 360*unitrand() ;
    return (r*cos(phi), r*sin(phi), z) ;
}

surface sphere=surface(genellfunc((R,R,R),(0,0,0),0,(0,0,1)),(0,0),(pi,2pi),32,64,Spline);
draw(sphere,white+opacity(0.1),render(merge=true));

for(int i=0; i < 100; ++i) {

    triple r = 0.1 * R * (unitrand(),unitrand(),unitrand());
    real Rmax = R - max(r.x,r.y,r.z);
    triple c = Rmax * (unitrand())**(1/3) * genunit();
    real omega = 360*unitrand();
    triple n = genunit();

    surface ell=surface(genellfunc(r,c,omega,n),(0,0),(pi,2pi),8,16,Spline);
    draw(ell,rgb(unitrand(),unitrand(),unitrand())+opacity(1),render(merge=true));

}

for(int i=0; i < 100; ++i) {

    real p = 10.**(unitrand()-0.5);
    triple r = 0.1 * R * (unitrand(),unitrand(),unitrand());
    real Rmax = R - max(r.x,r.y,r.z);
    triple c = Rmax * (unitrand())**(1/3) * genunit();
    real omega = 360*unitrand();
    triple n = genunit();

    surface supsph=surface(gensupsphfunc(p,r,c,omega,n),(0,0),(pi,2pi),8,16,Spline);
    draw(supsph,rgb(unitrand(),unitrand(),unitrand())+opacity(1),render(merge=true));

}
