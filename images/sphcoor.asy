settings.outformat = "html";
settings.render = -1;

// settings.outformat ="svg";
// settings.render = 0;
// // Hint: render svg format and then convert to png by "magick -density 300 file.svg file.png"

settings.outformat = "pdf";
settings.render = 8;
settings.prc = true;

import three;
import math;
texpreamble("\usepackage{bm}");

defaultpen(fontsize(18pt));

size(300,0);

pen thickp=linewidth(0.5mm);
real radius=1, theta=37, phi=60, psi=30;

triple e1=(1.,0.,0.);
triple e2=(0.,1.,0.);
triple e3=(0.,0.,1.);

triple v1=rotate(phi,e3)*rotate(theta,e2)*e1;
triple v2=rotate(phi,e3)*rotate(theta,e2)*e2;
triple v3=rotate(phi,e3)*rotate(theta,e2)*e3;

triple u1=rotate(phi,e3)*rotate(theta,e2)*rotate(psi,e3)*e1;
triple u2=rotate(phi,e3)*rotate(theta,e2)*rotate(psi,e3)*e2;
triple u3=rotate(phi,e3)*rotate(theta,e2)*rotate(psi,e3)*e3;

currentprojection=orthographic((4,2,1.5));

draw(octant1,material(palegray+opacity(0.25),shininess=0.5));

pen p=black+linewidth(1.2pt)+fontsize(18pt);
draw(Label("$\underline{e}_1$",1),O--X,p,Arrow3);
draw(Label("$\underline{e}_2$",1),O--Y,p,Arrow3);
draw(Label("$\underline{e}_3$",1),O--Z,p,Arrow3);

label("$\rm O$",(0,0,0),-1.5Y-0X);

triple Q=radius*dir(theta,phi);

draw(Q--Q+0.5*v1,dashed+red);
draw(Q--Q+0.5*v2,dashed+heavygreen);
draw(O--Q,dashed+black);

pen p=blue+linewidth(1.2pt)+fontsize(18pt);
draw(Label("$\underline{e}'_1$",1),Q--Q+u1,p,Arrow3);
draw(Label("$\underline{e}'_2$",1),Q--Q+u2,p,Arrow3);
draw(Label("$\underline{e}'_3$",1),Q--Q+u3,p,Arrow3);

// dot("$(x,y,z)$",Q);
draw(Q--(Q.x,Q.y,0),dashed+blue);
draw(O--radius*dir(90,phi),dashed+blue);
draw((0,0,Q.z)--Q,dashed+blue);
draw("$\theta$",arc(O,0.15*Z,0.15Q),align=2*dir(theta/2,phi),Arrow3);
draw("$\phi$",arc(O,0.15*X,0.15*dir(90,phi)),align=5*dir(90,phi/3)+Z,Arrow3);
draw("$\psi$",arc(Q,Q+0.4*v1,Q+0.4*u1),Arrow3);
draw("$\psi$",arc(Q,Q+0.4*v2,Q+0.4*u2),Arrow3);

// Spherical octant
real r=sqrt(Q.x^2+Q.y^2);
draw(arc((0,0,Q.z),(r,0,Q.z),(0,r,Q.z)),dashed+heavygreen);
draw(arc(O,radius*Z,radius*dir(90,phi)),dashed+red);
draw(arc(O,radius*Z,radius*X),thickp);
draw(arc(O,radius*Z,radius*Y),thickp);
draw(arc(O,radius*X,radius*Y),thickp);

// draw("$\bm{r}$",O--Q,align=2*dir(90,phi),Arrow3,DotMargin3);
