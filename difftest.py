import numpy as np
import matplotlib.pyplot as plt

#finite difference derivatives, for a space curve with arc length parametrisation

def threept_fd1(xm1,x,xp1):
    h2=np.linalg.norm(xp1-x)
    h1=np.linalg.norm(x-xm1)
    d1=h2*xm1/(h1*(h1+h2))
    d2=(h1-h2)*x/(h1*h2)
    d3=h1*xp1/(h2*(h1+h2))
    return -d1-d2+d3
   
def threept_fd2(xm1,x,xp1):
    h2=np.linalg.norm(xp1-x)
    h1=np.linalg.norm(x-xm1)
    return 2*(h2*xm1-(h1+h2)*x+h1*xp1)/(h1*h2*(h1+h2))
    
def fivept_fd1(xm2,xm1,x,xp1,xp2):
    h1=np.linalg.norm(xm1-xm2)
    h2=np.linalg.norm(x-xm1)
    h3=np.linalg.norm(xp1-x)
    h4=np.linalg.norm(xp2-xp1)
    H1=h1+h2+h3
    H2=h1+h2+h3+h4
    d1=h2*h3*(h3+h4)*xm2/(h1*(h1+h2)*H1*H2)
    d2=(h1+h2)*h3*(h3+h4)*xm1/(h1*h2*(h2+h3)*(H2-h1))
    d3=((h1+2*h2)*h3*(h3+h4)-(h1+h2)*h2*(2*h3+h4))*x/((h1+h2)*h2*h3*(h3+h4))
    d4=(h1+h2)*h2*(h3+h4)*xp1/(H1*(h2+h3)*h3*h4)
    d5=h1*(h1+h2)*H1*xp2/(H2*(H2-h1)*(h3+h4)*h4)
    return d1-d2+d3+d4-d5
    
def fivept_fd2(xm2,xm1,x,xp1,xp2):
    h1=np.linalg.norm(xm1-xm2)
    h2=np.linalg.norm(x-xm1)
    h3=np.linalg.norm(xp1-x)
    h4=np.linalg.norm(xp2-xp1)
    H1=h1+h2+h3
    H2=h1+h2+h3+h4
    d1=(-2*h2*(2*h3+h4)+2*h3*(h3+h4))*xm2/(h1*(h1+h2)*(h1+h2+h3)*H2)
    d2=(2*(h1+h2)*(2*h3+h4)-2*h3*(h3+h4))*xm1/(h1*h2*(h2+h3)*(h2+h3+h4))
    d3=(2*h2*(h1+h2)-2*(h1+2*h2)*(2*h3+h4)+2*h3*(h3+h4))*x/((h1+h2)*h2*h3*(h3+h4))
    d4=(2*(h1+2*h2)*(h3+h4)-2*h2*(h1+h2))*xp1/((h1+h2+h3)*(h2+h3)*h3*h4)
    d5=(2*(h1+h2)*h2-2*(h1+2*h2)*h3)*xp2/(H2*(h2+h3+h4)*(h3+h4)*h4)
    return d1+d2+d3+d4+d5
    
