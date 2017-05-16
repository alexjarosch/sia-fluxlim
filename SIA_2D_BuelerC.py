'''
Created on May 16, 2017

@author: Alexander H. Jarosch

This script demonstrates the 2D version of the MUSCL-SuperBee SIA code based on

A. H. Jarosch, C. G. Schoof, and F. S. Anslow, 
"Restoring mass conservation to shallow ice flow models over complex terrain",
The Cryosphere, Vol. 7, Iss. 1, pp. 229-240, 201
http://www.the-cryosphere.net/7/229/2013/

by recreating the benchmark C in
Bueler, E., Lingle, C.S., Kallen-Brown, J.A., Covey, D.N. and Bowman, L.N.,
"Exact solutions and verification of numerical models for isothermal ice sheets."
Journal of Glaciology, 51(173), pp.291-306, 2005
https://doi.org/10.3189/172756505781829449

Copyright 2017 Alexander H. Jarosch

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

from __future__ import division, print_function
import numpy
import matplotlib.pyplot as plt
import time

def main():
    # Start the timer.
    tstart = time.time()
    
    # Ice parameters defined
    A = 1.0e-16
    n = 3.
    g = 9.81
    rho = 910.
    Gamma = 2.*A*(rho*g)**n / (n+2) # we introduce Gamma to shorten the equations.

    # forward time stepping stability criteria
    CFL = 0.124 #just beyond R. Hindmarsh's idea of 1/2(n+1)
    
    dx = 50000.
    dt = 1.
    t_total = 15208.
    Nt = t_total/dt

    secInYear = 31556926.
    
    # extend into 2D
    Lx = 800000.   # domain size
    Ly = Lx
    dy = dx
    
    # create horizontal vectors
    x = numpy.arange(-Lx, Lx+dx ,dx)
    y = numpy.arange(-Ly, Ly+dy ,dy)
    # create horizontal grid
    X, Y = numpy.meshgrid(x, y)
    R = numpy.sqrt(X**2. + Y**2.)

    # flat bed initial condition
    B = numpy.zeros(numpy.shape(X))
    # initial empty mass balance
    M_dot= numpy.zeros(numpy.shape(R))
    
    # make the surface equal to the bedrock, so no ice
    S = B

    H_init = S - B
    M_vol = numpy.sum(H_init[:])*dx*dy
    for t in range(int(Nt)+1):
        cfl_t = 0
        t_pass = t*dt

        # use time dependent mass balance
        HB_step = H_bueler(R, t_pass, n)
        if t_pass > 0.:
            M_dot = (5. / t_pass) * HB_step
        
        while cfl_t < dt:
            div_q, dt_cfl = diffusion_MUSCL_2D(S, B, Gamma, n, dx, dy, CFL)
            dt_use = numpy.fmin(dt_cfl,dt-cfl_t)
            cfl_t = cfl_t + dt_use

            S = S + (M_dot + div_q)*dt_use
            S = numpy.maximum(S,B)
            
            print('t[year]: %04.3f, dt_use[days]: %.04f' % (t_pass+cfl_t, dt_use*365))
         
        # check for mass conservation
        H_step = S - B
        Vol_step = numpy.sum(H_step[:])*dx*dy
        I_H = numpy.zeros(numpy.shape(H_step))
        I_H[H_step>0] = 1
        M_step = M_dot*I_H
        M_vol = M_vol + numpy.sum(M_step)*dx*dy
        print("t[year]: %.1f current ice volume %.4f" % (t_pass+cfl_t, Vol_step))
        print("t[year]: %.1f current MB volume %.4f" % (t_pass+cfl_t, M_vol))
        print("t[year]: %.1f Mass %.11f qm" % (t_pass+cfl_t, Vol_step-M_vol))
                                                        
    # print timing
    t_needed = (time.time() - tstart)
    print("it took: " + str(t_needed) + " seconds to solve the problem ;)")
    
    # Display the final result
    plt.figure()
    plt.imshow(S-B, interpolation='none', extent=[-Lx,Lx,-Ly,Ly])
    plt.title('ice thickness')
    plt.colorbar()

    S_final = B + H_bueler(R,15208.,n)

    plt.figure()
    plt.imshow(S-S_final, interpolation='none', extent=[-Lx,Lx,-Ly,Ly])
    plt.title('ice thickness error')
    plt.colorbar()
    
    ## calculate the error 
    H_exact = S_final - B
    Vol_exact = numpy.sum(H_exact[:])*dx*dy
    
    H_final = S - B
    Vol_final = numpy.sum(H_final[:])*dx*dy

    Vol_exact_abs = 3997940.*1e9    # volume according to Bueler paper

    err = (Vol_final-Vol_exact)/Vol_exact*100
    err_abs = (Vol_final-Vol_exact_abs)/Vol_exact_abs*100
    
    print('The cumulative numerical error is %f percent' % err)
    print('The cumulative absolute error is %f percent' % err_abs)

    plt.show()     
        
def diffusion_MUSCL_2D(S, B, Gamma, n, dx, dy, CFL):
    
    Ny, Nx = numpy.shape(S)
    
    k = numpy.arange(0,Ny)
    kp = numpy.hstack([numpy.arange(1,Ny),Ny-1])
    kpp = numpy.hstack([numpy.arange(2,Ny),Ny-1,Ny-1])
    km = numpy.hstack([0,numpy.arange(0,Ny-1)])
    kmm = numpy.hstack([0,0,numpy.arange(0,Ny-2)])
    l = numpy.arange(0,Nx)
    lp = numpy.hstack([numpy.arange(1,Nx),Nx-1])
    lpp = numpy.hstack([numpy.arange(2,Nx),Nx-1,Nx-1])
    lm = numpy.hstack([0,numpy.arange(0,Nx-1)])
    lmm = numpy.hstack([0,0,numpy.arange(0,Nx-2)])
    
    # calculate ice thickness
    H = S-B
    
    ### all the k components
    # MUSCL scheme UP
    r_k_up_m = (H[numpy.ix_(k,l)]-H[numpy.ix_(km,l)])/(H[numpy.ix_(kp,l)]-H[numpy.ix_(k,l)])
    H_k_up_m = H[numpy.ix_(k,l)] + 0.5 * phi(r_k_up_m)*(H[numpy.ix_(kp,l)]-H[numpy.ix_(k,l)])
    r_k_up_p = (H[numpy.ix_(kp,l)] - H[numpy.ix_(k,l)])/(H[numpy.ix_(kpp,l)] - H[numpy.ix_(kp,l)])
    H_k_up_p = H[numpy.ix_(kp,l)] - 0.5 * phi(r_k_up_p)*(H[numpy.ix_(kpp,l)] - H[numpy.ix_(kp,l)])
    
    # calculate the slope gradient;
    s_k_grad_up = (((S[numpy.ix_(k,lp)] - S[numpy.ix_(k,lm)] + S[numpy.ix_(kp,lp)] - S[numpy.ix_(kp,lm)])/(4*dx))**2. + ((S[numpy.ix_(kp,l)] - S[numpy.ix_(k,l)])/dy)**2.)**((n-1.)/2.)
    D_k_up_m = Gamma * H_k_up_m**(n+2.) * s_k_grad_up
    D_k_up_p = Gamma * H_k_up_p**(n+2.) * s_k_grad_up
    
    D_k_up_min = numpy.fmin(D_k_up_m,D_k_up_p)
    D_k_up_max = numpy.fmax(D_k_up_m,D_k_up_p)
    D_k_up = numpy.zeros(numpy.shape(H))
    # include the local slope to identify upstream
    D_k_up[numpy.logical_and(S[numpy.ix_(kp,l)]<=S[numpy.ix_(k,l)],H_k_up_m<=H_k_up_p)] = D_k_up_min[numpy.logical_and(S[numpy.ix_(kp,l)]<=S[numpy.ix_(k,l)],H_k_up_m<=H_k_up_p)]
    D_k_up[numpy.logical_and(S[numpy.ix_(kp,l)]<=S[numpy.ix_(k,l)],H_k_up_m>H_k_up_p)] = D_k_up_max[numpy.logical_and(S[numpy.ix_(kp,l)]<=S[numpy.ix_(k,l)],H_k_up_m>H_k_up_p)]
    D_k_up[numpy.logical_and(S[numpy.ix_(kp,l)]>S[numpy.ix_(k,l)],H_k_up_m<=H_k_up_p)] = D_k_up_max[numpy.logical_and(S[numpy.ix_(kp,l)]>S[numpy.ix_(k,l)],H_k_up_m<=H_k_up_p)]
    D_k_up[numpy.logical_and(S[numpy.ix_(kp,l)]>S[numpy.ix_(k,l)],H_k_up_m>H_k_up_p)] = D_k_up_min[numpy.logical_and(S[numpy.ix_(kp,l)]>S[numpy.ix_(k,l)],H_k_up_m>H_k_up_p)]
    
    # MUSCL scheme DOWN
    r_k_dn_m = (H[numpy.ix_(km,l)]-H[numpy.ix_(kmm,l)])/(H[numpy.ix_(k,l)]-H[numpy.ix_(km,l)])
    H_k_dn_m = H[numpy.ix_(km,l)] + 0.5 * phi(r_k_dn_m)*(H[numpy.ix_(k,l)]-H[numpy.ix_(km,l)])
    r_k_dn_p = (H[numpy.ix_(k,l)] - H[numpy.ix_(km,l)])/(H[numpy.ix_(kp,l)] - H[numpy.ix_(k,l)])
    H_k_dn_p = H[numpy.ix_(k,l)] - 0.5 * phi(r_k_dn_p)*(H[numpy.ix_(kp,l)] - H[numpy.ix_(k,l)])
    
    # calculate the slope gradient;
    s_k_grad_dn = (((S[numpy.ix_(km,lp)] - S[numpy.ix_(km,lm)] + S[numpy.ix_(k,lp)] - S[numpy.ix_(k,lm)])/(4*dx))**2. + ((S[numpy.ix_(k,l)] - S[numpy.ix_(km,l)])/dy)**2.)**((n-1.)/2.)
    D_k_dn_m = Gamma * H_k_dn_m**(n+2.) * s_k_grad_dn
    D_k_dn_p = Gamma * H_k_dn_p**(n+2.) * s_k_grad_dn
    
    D_k_dn_min = numpy.fmin(D_k_dn_m,D_k_dn_p)
    D_k_dn_max = numpy.fmax(D_k_dn_m,D_k_dn_p)
    D_k_dn = numpy.zeros(numpy.shape(H))
    # include the local slope to identify upstream
    D_k_dn[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(km,l)],H_k_dn_m<=H_k_dn_p)] = D_k_dn_min[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(km,l)],H_k_dn_m<=H_k_dn_p)]
    D_k_dn[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(km,l)],H_k_dn_m>H_k_dn_p)] = D_k_dn_max[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(km,l)],H_k_dn_m>H_k_dn_p)]
    D_k_dn[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(km,l)],H_k_dn_m<=H_k_dn_p)] = D_k_dn_max[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(km,l)],H_k_dn_m<=H_k_dn_p)]
    D_k_dn[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(km,l)],H_k_dn_m>H_k_dn_p)] = D_k_dn_min[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(km,l)],H_k_dn_m>H_k_dn_p)]
    
    ### all the l components
    # MUSCL scheme UP
    r_l_up_m = (H[numpy.ix_(k,l)]-H[numpy.ix_(k,lm)])/(H[numpy.ix_(k,lp)]-H[numpy.ix_(k,l)])
    H_l_up_m = H[numpy.ix_(k,l)] + 0.5 * phi(r_l_up_m)*(H[numpy.ix_(k,lp)]-H[numpy.ix_(k,l)])
    r_l_up_p = (H[numpy.ix_(k,lp)] - H[numpy.ix_(k,l)])/(H[numpy.ix_(k,lpp)] - H[numpy.ix_(k,lp)])
    H_l_up_p = H[numpy.ix_(k,lp)] - 0.5 * phi(r_l_up_p)*(H[numpy.ix_(k,lpp)] - H[numpy.ix_(k,lp)])
    
    # calculate the slope gradient;
    s_l_grad_up = (((S[numpy.ix_(kp,l)] - S[numpy.ix_(km,l)] + S[numpy.ix_(kp,lp)] - S[numpy.ix_(km,lp)])/(4*dy))**2. + ((S[numpy.ix_(k,lp)] - S[numpy.ix_(k,l)])/dx)**2.)**((n-1.)/2.)
    D_l_up_m = Gamma * H_l_up_m**(n+2.) * s_l_grad_up
    D_l_up_p = Gamma * H_l_up_p**(n+2.) * s_l_grad_up
    
    D_l_up_min = numpy.fmin(D_l_up_m,D_l_up_p)
    D_l_up_max = numpy.fmax(D_l_up_m,D_l_up_p)
    D_l_up = numpy.zeros(numpy.shape(H))
    # include the local slope to identify upstream
    D_l_up[numpy.logical_and(S[numpy.ix_(k,lp)]<=S[numpy.ix_(k,l)],H_l_up_m<=H_l_up_p)] = D_l_up_min[numpy.logical_and(S[numpy.ix_(k,lp)]<=S[numpy.ix_(k,l)],H_l_up_m<=H_l_up_p)]
    D_l_up[numpy.logical_and(S[numpy.ix_(k,lp)]<=S[numpy.ix_(k,l)],H_l_up_m>H_l_up_p)] = D_l_up_max[numpy.logical_and(S[numpy.ix_(k,lp)]<=S[numpy.ix_(k,l)],H_l_up_m>H_l_up_p)]
    D_l_up[numpy.logical_and(S[numpy.ix_(k,lp)]>S[numpy.ix_(k,l)],H_l_up_m<=H_l_up_p)] = D_l_up_max[numpy.logical_and(S[numpy.ix_(k,lp)]>S[numpy.ix_(k,l)],H_l_up_m<=H_l_up_p)]
    D_l_up[numpy.logical_and(S[numpy.ix_(k,lp)]>S[numpy.ix_(k,l)],H_l_up_m>H_l_up_p)] = D_l_up_min[numpy.logical_and(S[numpy.ix_(k,lp)]>S[numpy.ix_(k,l)],H_l_up_m>H_l_up_p)]
    
    # MUSCL scheme DOWN
    r_l_dn_m = (H[numpy.ix_(k,lm)]-H[numpy.ix_(k,lmm)])/(H[numpy.ix_(k,l)]-H[numpy.ix_(k,lm)])
    H_l_dn_m = H[numpy.ix_(k,lm)] + 0.5 * phi(r_l_dn_m)*(H[numpy.ix_(k,l)]-H[numpy.ix_(k,lm)])
    r_l_dn_p = (H[numpy.ix_(k,l)] - H[numpy.ix_(k,lm)])/(H[numpy.ix_(k,lp)] - H[numpy.ix_(k,l)])
    H_l_dn_p = H[numpy.ix_(k,l)] - 0.5 * phi(r_l_dn_p)*(H[numpy.ix_(k,lp)] - H[numpy.ix_(k,l)])
    
    # calculate the slope gradient;
    s_l_grad_dn = (((S[numpy.ix_(kp,lm)] - S[numpy.ix_(km,lm)] + S[numpy.ix_(kp,l)] - S[numpy.ix_(km,l)])/(4*dy))**2. + ((S[numpy.ix_(k,l)] - S[numpy.ix_(k,lm)])/dx)**2.)**((n-1.)/2.)
    D_l_dn_m = Gamma * H_l_dn_m**(n+2.) * s_l_grad_dn
    D_l_dn_p = Gamma * H_l_dn_p**(n+2.) * s_l_grad_dn
    
    D_l_dn_min = numpy.fmin(D_l_dn_m,D_l_dn_p)
    D_l_dn_max = numpy.fmax(D_l_dn_m,D_l_dn_p)
    D_l_dn = numpy.zeros(numpy.shape(H))
    # include the local slope to identify upstream
    D_l_dn[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(k,lm)],H_l_dn_m<=H_l_dn_p)] = D_l_dn_min[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(k,lm)],H_l_dn_m<=H_l_dn_p)]
    D_l_dn[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(k,lm)],H_l_dn_m>H_l_dn_p)] = D_l_dn_max[numpy.logical_and(S[numpy.ix_(k,l)]<=S[numpy.ix_(k,lm)],H_l_dn_m>H_l_dn_p)]
    D_l_dn[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(k,lm)],H_l_dn_m<=H_l_dn_p)] = D_l_dn_max[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(k,lm)],H_l_dn_m<=H_l_dn_p)]
    D_l_dn[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(k,lm)],H_l_dn_m>H_l_dn_p)] = D_l_dn_min[numpy.logical_and(S[numpy.ix_(k,l)]>S[numpy.ix_(k,lm)],H_l_dn_m>H_l_dn_p)]
    
    # check the cfl condition
    dt_cfl = CFL * min(dx**2.,dy**2.) / max(max(max(abs(D_k_up.flatten())),max(abs(D_k_dn.flatten()))),max(max(abs(D_l_up.flatten())),max(abs(D_l_dn.flatten()))))
    
    # calculate final diffusion term
    div_k = (D_k_up * (S[numpy.ix_(kp,l)] - S[numpy.ix_(k,l)])/dy - D_k_dn * (S[numpy.ix_(k,l)] - S[numpy.ix_(km,l)])/dy)/dy
    div_l = (D_l_up * (S[numpy.ix_(k,lp)] - S[numpy.ix_(k,l)])/dx - D_l_dn * (S[numpy.ix_(k,l)] - S[numpy.ix_(k,lm)])/dx)/dx
    
    div_back = div_k + div_l
    
    return div_back, dt_cfl
    
def phi(r):
    
    # Koren
    # val_phi = numpy.fmax(0,numpy.fmin(numpy.fmin((2.*r),(2.+r)),2.))
    
    # superbee
    val_phi = numpy.fmax(0,numpy.fmin(numpy.fmin(2.*r,1.),numpy.fmin(r,2.)))

    return val_phi

def H_bueler(R,t_mb,n):
    # dome height according to Bueler et al. 2005 (Journal of Glaciology, Vol. 51, No. 173), test c
    lambda_B = 5.
    H_0_B = 3600.
    R_0_B = 750000.
    t_0_B = 15208.
    alpha_B = (2.-(n+1.)*lambda_B)/(5.*n+3.)
    beta_B = (1.+(2.*n+1.)*lambda_B)/(5.*n+3.)

    H_B = numpy.zeros(numpy.shape(R))

    if t_mb > 0.:
        H_B = H_0_B*(t_mb/t_0_B)**(-alpha_B) * (1.-((t_mb/t_0_B)**(-beta_B) * (R/R_0_B))**((n+1.)/n))**(n/(2.*n+1.))
        H_B[numpy.isnan(H_B)] = 0

    return H_B

if __name__ == '__main__':
    main()
