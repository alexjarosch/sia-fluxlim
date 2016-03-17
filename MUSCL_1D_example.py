'''
Created on Dec 14, 2012
@author: Alexander H. Jarosch

This script re-creates Figure 3 in the paper:
Numerical mass conservation issues in shallow ice models of mountain glaciers: the use of flux limiters and a benchmark

by

Alexander H. Jarosch, Christian G. Schoof, and Faron S. Anslow
for review in The Cryosphere Discussions 2012 (tc-2012-143)

Copyright 2012 Alexander H. Jarosch

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
'''

import numpy
import matplotlib.pyplot as plt

def main():
    
    # These parameters are defined in section 7.1
    A = 1e-16
    n = 3.
    g = 9.81
    rho = 910.
    Gamma = 2.*A*(rho*g)**n / (n+2) # we introduce Gamma to shorten the equations.
    mdot_0 = 2.
    x_m = 20000.
    x_s = 7000.
    b_0 = 500.
    c_stab = 0.165
    
    dx = 200.
    dt = 1.0
    t_total = 50000.
    Nt = t_total/dt
    
    # create a space vector
    x = numpy.arange(0.,x_m + x_m/2. + dx,dx)
    # define the mass balance vector
    m_dot = accumulation(x, mdot_0, n, x_m)
    # define bed vector. Here we use capital B instead of b.
    B = numpy.zeros(len(x))
    B[x<x_s]=b_0
    
    # set the intial ice surface S to the bed elevation. Again we use capital S instead of s.
    S = B
    Nx = len(S)
    
    # finite difference indixes. k is used in this 1D example. 
    # Note that python starts counting vector entries with 0 instead of 1, as e.g. MATLAB does
    k = numpy.arange(0,Nx)
    kp = numpy.hstack([numpy.arange(1,Nx),Nx-1])
    kpp = numpy.hstack([numpy.arange(2,Nx),Nx-1,Nx-1])
    km = numpy.hstack([0,numpy.arange(0,Nx-1)])
    kmm = numpy.hstack([0,0,numpy.arange(0,Nx-2)])
    
    # MUSCL scheme loop
    for t in range(int(Nt)+1):
        stab_t = 0.
        while stab_t < dt:
            H = S - B
            
            # MUSCL scheme up. "up" denotes here the k+1/2 flux boundary
            r_up_m = (H[k]-H[km])/(H[kp]-H[k])                      # Eq. 27
            H_up_m = H[k] + 0.5 * phi(r_up_m)*(H[kp]-H[k])          # Eq. 23
            r_up_p = (H[kp]-H[k])/(H[kpp]-H[kp])                    # Eq. 27, now k+1 is used instead of k
            H_up_p = H[kp] - 0.5 * phi(r_up_p)*(H[kpp]-H[kp])       # Eq. 24
            
            # surface slope gradient
            s_grad_up = ((S[kp]-S[k])**2. / dx**2.)**((n-1.)/2.)
            D_up_m = Gamma * H_up_m**(n+2.) * s_grad_up             # like Eq. 30, now using Eq. 23 instead of Eq. 24
            D_up_p = Gamma * H_up_p**(n+2.) * s_grad_up             # Eq. 30
            
            D_up_min = numpy.minimum(D_up_m,D_up_p);                # Eq. 31
            D_up_max = numpy.maximum(D_up_m,D_up_p);                # Eq. 32
            D_up = numpy.zeros(Nx)
            
            # Eq. 33
            D_up[numpy.logical_and(S[kp]<=S[k],H_up_m<=H_up_p)] = D_up_min[numpy.logical_and(S[kp]<=S[k],H_up_m<=H_up_p)]
            D_up[numpy.logical_and(S[kp]<=S[k],H_up_m>H_up_p)] = D_up_max[numpy.logical_and(S[kp]<=S[k],H_up_m>H_up_p)]
            D_up[numpy.logical_and(S[kp]>S[k],H_up_m<=H_up_p)] = D_up_max[numpy.logical_and(S[kp]>S[k],H_up_m<=H_up_p)]
            D_up[numpy.logical_and(S[kp]>S[k],H_up_m>H_up_p)] = D_up_min[numpy.logical_and(S[kp]>S[k],H_up_m>H_up_p)]

            # MUSCL scheme down. "down" denotes here the k-1/2 flux boundary
            r_dn_m = (H[km]-H[kmm])/(H[k]-H[km])
            H_dn_m = H[km] + 0.5 * phi(r_dn_m)*(H[k]-H[km])
            r_dn_p = (H[k]-H[km])/(H[kp]-H[k])
            H_dn_p = H[k] - 0.5 * phi(r_dn_p)*(H[kp]-H[k])
            
            # calculate the slope gradient
            s_grad_dn = ((S[k]-S[km])**2. / dx**2.)**((n-1.)/2.)
            D_dn_m = Gamma * H_dn_m**(n+2.) * s_grad_dn
            D_dn_p = Gamma * H_dn_p**(n+2.) * s_grad_dn
            
            D_dn_min = numpy.minimum(D_dn_m,D_dn_p);
            D_dn_max = numpy.maximum(D_dn_m,D_dn_p);
            D_dn = numpy.zeros(Nx)
            
            D_dn[numpy.logical_and(S[k]<=S[km],H_dn_m<=H_dn_p)] = D_dn_min[numpy.logical_and(S[k]<=S[km],H_dn_m<=H_dn_p)]
            D_dn[numpy.logical_and(S[k]<=S[km],H_dn_m>H_dn_p)] = D_dn_max[numpy.logical_and(S[k]<=S[km],H_dn_m>H_dn_p)]
            D_dn[numpy.logical_and(S[k]>S[km],H_dn_m<=H_dn_p)] = D_dn_max[numpy.logical_and(S[k]>S[km],H_dn_m<=H_dn_p)]
            D_dn[numpy.logical_and(S[k]>S[km],H_dn_m>H_dn_p)] = D_dn_min[numpy.logical_and(S[k]>S[km],H_dn_m>H_dn_p)]
            
            dt_stab = c_stab * dx**2. / max(max(abs(D_up)),max(abs(D_dn)))      # Eq. 37
            dt_use = min(dt_stab,dt-stab_t)
            stab_t = stab_t + dt_use
            
            #explicit time stepping scheme
            div_q = (D_up * (S[kp] - S[k])/dx - D_dn * (S[k] - S[km])/dx)/dx    # Eq. 36
            S = S[k] + (m_dot + div_q)*dt_use                                   # Eq. 35
            
            S = numpy.maximum(S,B)                                              # Eq. 7
        
            print 'MUSCL Year %d, dt_use %f' % (t,dt_use)
    
    
        if numpy.mod(t,1000.)==0.0:
            plt.plot(x/1000,S,'-b')
            
    # calculate the volume difference
    p1, = plt.plot(x/1000,S,'-b')
    H = S-B
    vol_muscl = numpy.trapz(H,x)

    # upstream loop, which performs the same benchmark, just with a upstream scheme
    S = B     
    for t in range(int(Nt)+1):
        stab_t = 0.
        while stab_t < dt:
            H = S - B
            
            H_up = 0.5 * ( H[kp] + H[k] )
            H_dn = 0.5 * ( H[k] + H[km] )
            # applying Eq. (61) to the scheme
            Hk = H[k]
            Hkp = H[kp]
            Hkm = H[km]
            H_upstream_up = numpy.zeros(len(k))
            H_upstream_up[S[kp]>S[k]] = Hkp[S[kp]>S[k]]
            H_upstream_up[S[k]>=S[kp]] = Hk[S[k]>=S[kp]]
            
            H_upstream_dn = numpy.zeros(len(k))
            H_upstream_dn[S[k]>S[km]] = Hk[S[k]>S[km]]
            H_upstream_dn[S[km]>=S[k]] = Hkm[S[km]>=S[k]]

            s_grad_up = ((S[kp]-S[k])**2. / dx**2.)**((n-1.)/2.)
            s_grad_dn = ((S[k]-S[km])**2. / dx**2.)**((n-1.)/2.)
            D_up = Gamma * H_up**(n+1) * H_upstream_up * s_grad_up
            D_dn = Gamma * H_dn**(n+1) * H_upstream_dn * s_grad_dn          
            
            dt_stab = c_stab * dx**2. / max(max(abs(D_up)),max(abs(D_dn)))
            dt_use = min(dt_stab,dt-stab_t)
            stab_t = stab_t + dt_use
            
            #explicit time stepping scheme
            div_q = (D_up * (S[kp] - S[k])/dx - D_dn * (S[k] - S[km])/dx)/dx
            S = S[k] + (m_dot + div_q)*dt_use
            
            S = numpy.maximum(S,B)    
        
            print 'Upstream Year %d, dt_use %f' % (t,dt_use)
    
    
        if numpy.mod(t,1000.)==0.0:
            plt.plot(x/1000,S,'-g')
            
    # calculate the volume difference
    p2, = plt.plot(x/1000,S,'-g')
    H = S-B
    vol_upstream = numpy.trapz(H,x)
    
    # M2 loop, which performs the same benchmark, just with a M2 scheme
    S = B     
    for t in range(int(Nt)+1):
        stab_t = 0.
        while stab_t < dt:
            H = S - B
            
            H_up = 0.5 * ( H[kp] + H[k] )
            H_dn = 0.5 * ( H[k] + H[km] )
                        
            s_grad_up = ((S[kp]-S[k])**2. / dx**2.)**((n-1.)/2.)
            s_grad_dn = ((S[k]-S[km])**2. / dx**2.)**((n-1.)/2.)
            D_up = Gamma * H_up**(n+2) * s_grad_up
            D_dn = Gamma * H_dn**(n+2) * s_grad_dn            
            
            dt_stab = c_stab * dx**2. / max(max(abs(D_up)),max(abs(D_dn)))
            dt_use = min(dt_stab,dt-stab_t)
            stab_t = stab_t + dt_use
            
            #explicit time stepping scheme
            div_q = (D_up * (S[kp] - S[k])/dx - D_dn * (S[k] - S[km])/dx)/dx
            S = S[k] + (m_dot + div_q)*dt_use
            
            S = numpy.maximum(S,B)    
        
            print 'M2 Year %d, dt_use %f' % (t,dt_use)
    
    
        if numpy.mod(t,1000.)==0.0:
            plt.plot(x/1000,S,'-r')
            
    # calculate the volume difference
    p3, = plt.plot(x/1000,S,'-r')
    H = S-B
    vol_M2 = numpy.trapz(H,x)
                
    # create the explicit steady state solution form section 6, which we call s here
    s_x_s_x_m = s_eval_x_s_x_m(x,x_s,x_m,n,A,mdot_0,rho,g)
    s_x_s = s_eval_x_s(x,x_s,x_m,n,A,mdot_0,rho,g,b_0)
    # combine the solutions
    s = s_x_s+b_0
    s[x >= x_s] = s_x_s_x_m[x >= x_s]
    # correct s
    s[x>x_m] = 0.
    
    h = s-B
    vol_exact = numpy.trapz(h,x)
    vol_err_muscl = (vol_muscl-vol_exact)/vol_exact*100
    vol_err_upstream = (vol_upstream-vol_exact)/vol_exact*100
    vol_err_M2 = (vol_M2-vol_exact)/vol_exact*100
    
    print "vol exact: %e" % vol_exact
    print "vol muscl: %e" % vol_muscl
    print "vol upstream: %e" % vol_upstream
    print "vol M2: %e" % vol_M2
    print "err muscl %0.3f" % vol_err_muscl
    print "err upstream: %0.3f" % vol_err_upstream
    print "err M2 %0.3f" % vol_err_M2
    
    p4, = plt.plot(x/1000,B,'-k',linewidth=2)
    p5, = plt.plot(x/1000,s,'-',linewidth=3,color='#ff7800')
    plt.xlabel('x [km]')
    plt.ylabel('z [m]')
    plt.title('50000 years evolution')
    plt.legend([p1, p2, p3, p4, p5], ["MUSCL superbee", "Upstream", "M2", "bed", "Eqs. (56) and (57)"])
    plt.show()

    
    
def accumulation(x,mdot_0,n,x_m):
    # Eq. 54
    mdot = ((n*mdot_0)/(x_m**(2.*n-1.)))*x**(n-1.)*(abs(x_m-x)**(n-1.))*(x_m-2.*x)
    
    mdot[x>x_m] = 0.
    
    return mdot 
    
def s_eval_x_s_x_m(x,x_s,x_m,n,A,mdot_0,rho,g):
    # Eq. 56
    s_x_s_x_m = (((2.*n+2.)*(n+2.)**(1./n)*mdot_0**(1./n))/(2.**(1./n)*6*n*A**(1./n)*rho*g*x_m**((2.*n-1)/n))*(x_m+2.*x)*(x_m-x)**2.)**(n/(2.*n+2.))
    
    return s_x_s_x_m
    
def s_eval_x_s(x,x_s,x_m,n,A,mdot_0,rho,g,b_0):
    # Eq. 58 
    h_splus = (((2.*n+2.)*(n+2.)**(1./n)*mdot_0**(1./n))/(2.**(1./n)*6*n*A**(1./n)*rho*g*x_m**((2.*n-1.)/n))*(x_m+2.*x_s)*(x_m-x_s)**2.)**(n/(2.*n+2.))
    # Eq. 59
    h_sminus = numpy.maximum(h_splus - b_0,0.)
    # Eq. 57
    h_back = (h_sminus**((2.*n+2.)/n)-h_splus**((2.*n+2.)/n)+((2.*n+2.)*(n+2.)**(1./n)*mdot_0**(1./n))/(2.**(1./n)*6*n*A**(1./n)*rho*g*x_m**((2.*n-1.)/n))*(x_m+2.*x)*(x_m-x)**2.)**(n/(2.*n+2.))
    
    return h_back

def phi(r):
    # an overview of possible flux limiters can be found at https://en.wikipedia.org/wiki/Flux_limiter
    # Flux limiter function by Sweby 1984. For beta = 1.0 its equal to Roe's minmod limiter and for beta = 2.0 its equal to Roe's superbee limiter.
    # Values of beta between 1 and 2 keep the limiter second-order TVD
    beta = 2.0
    phi_sw = numpy.maximum(0, numpy.maximum(numpy.minimum(beta*r, 1.), numpy.minimum(r, beta)))

    return phi_sw

''' DEFINE which routine to run as the main '''

if __name__=="__main__":
    main()
