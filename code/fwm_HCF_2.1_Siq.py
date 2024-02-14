# -*- coding: utf-8 -*-
import numpy as np
import sys
from scipy import pi
from scipy import integrate
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tftb #need to install this
import sys
sys.path.append("/Users/davidgarcia/Documents/GitHub/froglib") #need to clone this repository to corect directory path
from froglib import *

'''
This is a code that computes pulse propagation in a Hollow Core Fiber (HCF) filled with a noble gas.
'''

'''
Constants
'''
c = 299792458.0 # m/s
mu_0 = 4.0*pi*1e-7 # N/A^2
eps0 = 1.0/(mu_0*c**2) # F/m

'''
Coefficients
rN_gas:
    Helper function for equations, it returns a constant (makes equations more compact)
chi3_gas:
    Gives third order nonlinearity of noble gasses in [m^2/V^2] [2]
    The temerature and pressure dependence needs to be verified
nSqr_gas:
    Returns the square of the refractive index of noble gasses [3] & it's first, second, and third deritivatives with respect to angular frequency w,
    these are used to calculate higher order disperion coefficients.
nSqr_SiO2 & nSqr_Ag:
    Returns square of refrative index for fused silica (Si02) [5] and silver (Ag), these are used for loss coefficients
alpha:
    Returns loss coefficient [4]
betas:
    Returns axial wavevector (beta) and it's w deritives known as higher order dispersion coefficients
    B = w*n/c is index of refraction in an optical waveguide [6]
GVM:
    Returns group velocity mismiatch coefficient in [s/m] [1]
gamma:
    Returns nonlinear coefficient in [1/(m*W)] [1]
'''
def rN_gas(p,T):
    T0 = 273.15 #in Kelvin
    p0 = 1.01325 #1 atm given in bar
    return (p*T0)/(p0*T)

def chi3_gas(gas, p, T=293.15):
    Chi3He = 3.43e-28 # at standard conditions
    Chi3 = {
        'He' : Chi3He,
        'Ne' : 1.8*Chi3He,
        'Ar' : 23.5*Chi3He,
        'Kr' : 64.0*Chi3He,
        'Xe' : 188.2*Chi3He,
            }
    #I do not know where this pressure and temperature dependence comes from
    return 4*rN_gas(p,T)*Chi3[gas]

def nSqr_gas(w, gas, p, T=293.15):
    rN = rN_gas(p,T)

    if gas == 'He':
        B1, C1, B2, C2 = 4977.77*1e-8, 28.54*1e-6, 1856.94*1e-8, 7.760*1e-3
    elif gas == 'Ne':
        B1, C1, B2, C2 = 9154.48*1e-8, 656.97*1e-6, 4018.63*1e-8, 5.728*1e-3
    elif gas == 'Ar':
        B1, C1, B2, C2 = 20332.29*1e-8, 206.12*1e-6, 34458.31*1e-8, 8.066*1e-3
    elif gas == 'Kr':
        B1, C1, B2, C2 = 26102.88*1e-8, 2.01*1e-6, 56946.82*1e-8, 10.043*1e-3
    elif gas == 'Xe':
        B1, C1, B2, C2 = 103701.61*1e-8, 12.75e3*1e-6, 31228.61*1e-8,  0.561*1e-3
    elif gas == 'Air':
        B1, C1, B2, C2 = 14926.44*1e-8, 19.36*1e-6, 41807.57*1e-8, 7.434*1e-3
    else:
        raise ValueError('gas unknown')

    lmd = (2*pi*c/w)*1e6 #the paper wants the wavelength in microns
    lmd2 = lmd**2 #the sellmeier equation wants lambda squared, we do it here for cleanliness

    nSqr = 1 + rN*( B1*lmd2/(lmd2 - C1) + B2*lmd2/(lmd2 - C2) ) #this is n^2_gas found using the sellmeier equation

    #To calculate higer order dispersion coefficiencts we need to compute the w deritives of the square of the refractive index of the gas
    #to do this we first find lambda derivitives and then convert these into w derivitives.
    dl1_nSqr = (-2*rN*lmd*( B1*C1*(C2 - lmd2)**2 + B2*C2*(C1 - lmd2)**2 )/( (C1 - lmd2)*(C2 - lmd2) )**2)*1e6 #dn^2_gas/dl, first lambda derivative of nSqr_gas
    dl2_nSqr = (2*rN*( B1*C1*(C1+3*lmd2)*(C2 - lmd2)**3 + B2*C2*(C2+3*lmd2)*(C1 - lmd2)**3 )/( (C1 - lmd2)*(-C2 + lmd2) )**3)*1e12 #second  lambda derivative of nSqr_gas
    dl3_nSqr = (-24*rN*lmd*( B1*C1*(C1 + lmd2)*(C2 - lmd2)**4 + B2*C2*(C2 + lmd2)*(C1 - lmd2)**4 )/( (C1 - lmd2)*(C2 - lmd2) )**4)*1e18 #third lambda derivative of nSqr_gas

    dw1_nSqr = - (2*pi*c/w**2)*dl1_nSqr #dn^2_gas/dw, first w derivative of nSqr_gas
    dw2_nSqr = (4*pi*c/w**3)*dl1_nSqr + (4*pi**2 *c**2 /w**4)*dl2_nSqr #second w derivative of nSqr_gas
    dw3_nSqr = - (12*pi*c/w**4)*dl1_nSqr - (24*pi**2 *c**2 /w**5)*dl2_nSqr - (8*pi**3 *c**3 /w**6)*dl3_nSqr #third w derivative of nSqr_gas

    return np.array([nSqr, dw1_nSqr, dw2_nSqr, dw3_nSqr])

def nSqr_SiO2(w): #from paper, only valid for T = 293.15
    lmd = (2*pi*c/w)*1e6 #the paper wants the wavelength in microns
    lmd2 = lmd**2 #the sellmeier equation wants lambda squared, we do it here for cleanliness
    nSqr = 1 + 0.6961663*lmd2/(lmd2-0.0684043**2)+ 0.4079426*lmd2/(lmd2-0.1162414**2) + 0.8974794*lmd2/(lmd2-9.896161**2)

    return nSqr

#need nSqr_Ag(w) function for losses in a silver cladded fiber which is what I currently have

def alpha(w, radius, p, gas, T=293.15): #loss term from paper
    lmd = 2*pi*c/w #wavelength in m
    unm = 2.405 # first zero of bessell function (fundamental fiber mode approximation)
    [nSqr_Gas, dw1_nSqr, dw2_nSqr, dw3_nSqr] = nSqr_gas(w,gas,p,T)
    nSqr_FS = nSqr_SiO2(w) #change this to silver
    vSqr = nSqr_FS/nSqr_Gas
    alpha_coeff = (unm*lmd/(2*pi))**2/(radius**3) * (vSqr+1)/np.sqrt(vSqr-1)

    return alpha_coeff

def betas(w, radius, p, gas, T=293.15):
    unm = 2.405 # first zero of bessell function (fundamental fiber mode approximation)
    [nSqr, dw1_nSqr, dw2_nSqr, dw3_nSqr] = nSqr_gas(w,gas,p,T)

    beta = np.sqrt( (w/c)**2 *nSqr - (unm/radius)**2)#axial wavevector
    dw1_beta = (1/beta)*( w*nSqr/c**2 + w**2 *dw1_nSqr/(2*c**2) ) #dB/dw first w derivative of beta
    dw2_beta = (1/beta)*( nSqr/c**2 + 2*w*dw1_nSqr/c**2 + w**2 *dw2_nSqr/(2*c**2) - dw1_beta**2 ) #d^2B/dw^2 second w derivative of beta
    dw3_beta = (1/beta)*( 3*dw1_nSqr/c**2 + 3*w *dw2_nSqr/c**2 + w**2 *dw3_nSqr/(2*c**2) - 3*dw1_beta*dw2_beta ) #d^3B/dw^3 third w derivative of beta

    return np.array([beta, dw1_beta, dw2_beta, dw3_beta])

def GVM(ang_freqs,radius,p,gas,T=293.15): #group velocity mismatch
    wj, wk = ang_freqs #you have to put the frequencies in order so that you get the right value
    [beta0_j, beta1_j, beta2_j, beta3_j] = betas(wj,radius,p,gas,T)
    [beta0_k, beta1_k, beta2_k, beta3_k] = betas(wk,radius,p,gas,T)
    return (beta1_j - beta1_k)

def gamma(w, radius, p, gas, T=293.15):
    Aeff = 1.5*radius**2 #I need to verify this
    [nSqr, dw1_nSqr, dw2_nSqr, dw3_nSqr] = nSqr_gas(w,gas,p,T)
    n2 = 3*chi3_gas(gas, p, T)/(4*eps0*c*nSqr) #intenisty dependent refractive index, see Boyd
    gamma = n2*w/(c*Aeff) #nonlinear coefficient, see Argawal
    return gamma

'''
get_params:
    Collects parameters at central frequencies for each pulse
    For now I use the value of the paramter at the central frequency only
lengths:
    Returns charasteritic lengths [1]
'''
def get_params(ang_freqs, w_w0, radius, pressure, gas, T=293.15):
    [wp,ws,wi] = ang_freqs #wp=pump w, ws=signal w, wi=idler w

    alphas = [alpha(w_w0+j, radius, pressure, gas, T) for j in [wp,ws,wi]]

    [beta0_p, beta1_p, beta2_p, beta3_p] = betas(w_w0+wp,radius,pressure,gas,T)
    [beta0_s, beta1_s, beta2_s, beta3_s] = betas(w_w0+ws,radius,pressure,gas,T)
    [beta0_i, beta1_i, beta2_i, beta3_i] = betas(w_w0+wi,radius,pressure,gas,T)

    db = beta0_s + beta0_i - 2*beta0_p

    d_sp = GVM([w_w0+ws,w_w0+wp],radius,pressure,gas,T)
    d_ip = GVM([w_w0+wi,w_w0+wp],radius,pressure,gas,T)

    gammas = [gamma(w_w0+i, radius, pressure, gas, T) for i in [wp,ws,wi] ]

    params = [gammas, db, [d_sp, d_ip], alphas, [beta0_p, beta1_p, beta2_p, beta3_p], [beta0_s, beta1_s, beta2_s, beta3_s], [beta0_i, beta1_i, beta2_i, beta3_i] ]

    return params

def lengths(ang_freqs, params, power, t_FWHM): #computes charasteritic lengths
    [wp,ws,wi] = ang_freqs
    [gammas, db, [d_sp, d_ip], alphas, [beta0_p, beta1_p, beta2_p, beta3_p], [beta0_s, beta1_s, beta2_s, beta3_s], [beta0_i, beta1_i, beta2_i, beta3_i] ] = params
    L_NL = 1/(gammas[0]*power[0]) #nonlinear length in m
    L_W = t_FWHM/np.abs([d_sp, d_ip]) #walkoff length in m
    L_D = t_FWHM**2/np.abs([beta2_p,beta2_s,beta2_i]) #GVD length in m
    L_T = t_FWHM**3/np.abs([beta3_p,beta3_s,beta3_i]) #TOD length in m

    return [L_NL, L_W, L_D, L_T]

'''
optimize_pressure:
    Finds pressure that gives smallest phase mismatch for central wavelengths
'''
def optimize_pressure(ang_freqs, w_w0, pressure_range, radius, gas_type, T=293.15, num_points = 1000):
    #num_points is subdivision of pressure_range
    press = np.linspace(pressure_range[0], pressure_range[1], num_points)
    db = [] #phase mismatch
    for ps in press:
        a = get_params(ang_freqs, 0, radius, ps, gas_type, T) #we want smallest phase mismatch at central wavelengths
        db.append(a[1])
    db = np.array(db)
    index_min = np.argmin(np.abs(db)) #index of smallest phase mismatch
    return get_params(ang_freqs, w_w0, radius, press[index_min], gas_type, T), press[index_min] #returns nonlinear parameters at optimal pressure and the optimal pressure

'''
Pulse Propagation Equations are coupled PDEs [1]
If we want to ignore dispersion we set higher order disperison coefficients to 0 in betas function

D, N, & RK4IP:
    This solves the coupled PDE's with an RK4 method in the interaction picture [7]
    D is dispersion operator with losses, N is nonlinear operator.
    Must be solved with a pulse in time space
'''
def D(ang_freqs, w_w0, params): #dispersion operator
    [wp,ws,wi] = ang_freqs
    [[gamma_p, gamma_s, gamma_i], db, [d_sp, d_ip], [alpha_p, alpha_s, alpha_i], [beta0_p, beta1_p, beta2_p, beta3_p], [beta0_s, beta1_s, beta2_s, beta3_s], [beta0_i, beta1_i, beta2_i, beta3_i] ] = params
    #only take what you need from params?
    Dp = 1j*beta2_p*w_w0**2/2 - 1j*beta3_p*w_w0**3/6 - alpha_p/2
    Ds = -1j*d_sp*w_w0 + 1j*beta2_s*w_w0**2/2 - 1j*beta3_s*w_w0**3/6 - alpha_s/2
    Di = -1j*d_ip*w_w0 + 1j*beta2_i*w_w0**2/2 - 1j*beta3_i*w_w0**3/6 - alpha_i/2

    return np.array([Dp,Ds,Di])

def N(z, A, params): #nonlinear operator
    [gamma_p, gamma_s, gamma_i], db = params[0:2]
    P = np.array([np.abs(i)**2 for i in A])
    Np = np.fft.ifft(gamma_p*np.fft.fft( 1j*( (P[0] + 2*(P[1] + P[2]) )*A[0] + np.fft.ifft(np.fft.fft( 2*A[1]*A[2]*np.conj(A[0]) )*np.exp(+1j*db*z)) ) ))
    Ns = np.fft.ifft(gamma_s*np.fft.fft( 1j*( (P[1] + 2*(P[0] + P[2]) )*A[1] + np.fft.ifft(np.fft.fft( A[0]*A[0]*np.conj(A[2]) )*np.exp(-1j*db*z)) ) ))
    Ni = np.fft.ifft(gamma_i*np.fft.fft( 1j*( (P[2] + 2*(P[0] + P[1]) )*A[2] + np.fft.ifft(np.fft.fft( A[0]*A[0]*np.conj(A[1]) )*np.exp(-1j*db*z)) ) ))

    return np.array([Np,Ns,Ni])

def RK4IP(ang_freqs, w_w0, z, A, dz, params, time_step):
    d = D(ang_freqs, w_w0, params)

    AI = np.fft.ifft( np.exp(dz*d/2)*np.fft.fft(A) )

    k1 = np.fft.ifft( np.exp(dz*d/2)*np.fft.fft(N(z, A, params)) )

    k2 = N(z+dz/2, AI+dz*k1/2, params)

    k3 = N(z+dz/2, AI+dz*k2/2, params)

    Ak4 = np.fft.ifft( np.exp(dz*d/2)*np.fft.fft(AI+dz*k3) ) #function we use to evaluate N in k4
    k4 = N(z+dz, Ak4, params)

    B = np.fft.ifft( np.exp(dz*d/2)*np.fft.fft(AI + (k1 + 2*k2 + 2*k3)*dz/6) )

    dA = B + dz*k4/6

    k5 = N(z+dz, dA, params)

    dA3 = B + (2*k4 + 3*k5)*dz/30

    err = np.sum(np.sqrt(integrate.simpson(np.abs(dA-dA3)**2, dx=time_step, even='avg')))

    return [dA,err]

'''
propagator:
    Propagates pulse propagation solver along the length of the fiber.
    RK4IP has adaptive step sizing [7]
'''

def propagator(L, ang_freqs, w_w0, A, params, time_step, z = 0):
    out_z = []
    out_A = []
    dz = 1e-4 #step size
    tol = 1e-9 #keep error in each adaptive step below a specified tolerance (for RK4IP)
    while z < L:
        out_z.append(z)
        out_A.append(A)
        A,err = RK4IP(ang_freqs, w_w0, z, A, dz, params, time_step) #half step
        # print([round(z*100,3),round(dz*100,3),err])
        z += dz
        dz = max([.5,min( 2,np.power(tol/err,1/4) )])*dz #adaptive step sizing for RK4IP method
    return np.array(out_z), np.array(out_A)

'''
main:
    define input pulses(energy, pulse duration), length/radius of fiber, gas type, finds optimal pressure, plots pulse propagation,...
    I'm currently comparing the output of this simulation to the experiment done in [8]
'''

def main():
    wavelengths = [390e-9,780e-9] #wavelength of pump and signal pulses in m
    [wp, ws] = 2*pi*c/np.array(wavelengths) #central angular frequency of pump and signal
    wi = 2*wp - ws #central angular frequency of idler
    ang_freqs = np.array([wp,ws,wi])
    length = .2 #length of fiber in m

    pressure_range = [0,2] #range of pressures in bar for optimize_pressure to scan over
    radius = 75e-6 #inner radius of fiber in m
    gas = 'Ar'
    T = 293.15 #temperature of fiber in K

    pulse_duration = 35e-15 #FWHM of intensity of pulses in s
    energy_in = np.array([.2,.1,0])*1e-3*.5 #energy of each pulse in J
    power_in = energy_in/pulse_duration #input powers in W

    #PULSE IN TIME DOMAIN
    t_start = -5*pulse_duration
    t_end = 5*pulse_duration
    num_samples = 150 #number of sample points for the pulse
    time_step = (t_end - t_start)/num_samples
    t = np.arange(t_start, t_end, time_step) #create list of points to sample the function at
    w_w0 = 2*pi*np.fft.fftfreq(t.shape[-1],time_step) #this is w-w0. It is affected by num_samples and the time_step.
    #IF w_w0+[wp,ws,wi] HAS NEGATIVE FREQUENCY COMPONENTS THEN THERE WILL BE ISSUS COMPUTING DISPERISON PARAMTERS, THEY AREN'T DEFINED WELL FOR 0 FREQUENCY. ADJUST w_w0 ACCORDINGLY
    l_l0 = np.flip(np.linspace(-150e-9, 150e-9, num=t.shape[-1])) #lambda-lambda0 to plot pulses, it's flipped to match the pulse given in w space

    #INPUT FIELDS
    fields_in = np.array([np.sqrt(p)*np.exp( -4*np.log(2)*(t/pulse_duration)**2 ) for p in power_in]).astype('cdouble')
    # fields_in[2] = (100e-3)*np.exp(0*t)

    # phase_mask = pi*np.heaviside(-w_w0,1)
    # fig_phase, ax_phase = plt.subplots(3,2)
    # ax_phase[0,0].plot(w_w0,phase_mask)
    # ax_phase[1,0].plot( np.fft.fftshift(w_w0), np.fft.fftshift(phase_mask) )
    # ax_phase[0,1].plot(l_l0, np.fft.fftshift(phase_mask) )
    # ax_phase[1,1].plot(l_l0, np.abs(fields_in[1]) )
    # ax_phase[2,0].plot(w_w0)
    # ax_phase[2,1].plot(l_l0)
    # plt.show()
    # fields_in[1] = np.fft.ifft( np.fft.fft( fields_in[1] )*np.exp(-1j*pi*np.heaviside(-w_w0,1)) ) #apply phase mask to pulse in freq space

    #OPTIMIZE PRESSURE & SET PARAMTERES
    params, opt_pressure = optimize_pressure(ang_freqs, w_w0, pressure_range, radius, gas, T)

    #SET PRESSURE & SET PARAMTERES
    # set_pressure = .5
    # params, opt_pressure = get_params(ang_freqs, w_w0, radius, set_pressure, gas, T), set_pressure

    # #PRINT PARAMETERS AT CENTRAL FREQUENCIES
    sample_params = get_params(ang_freqs, 0, radius, opt_pressure, gas, T)
    [gammas, db, [d_sp, d_ip], alphas, [beta0_p, beta1_p, beta2_p, beta3_p], [beta0_s, beta1_s, beta2_s, beta3_s], [beta0_i, beta1_i, beta2_i, beta3_i] ] = sample_params
    charasteritic_lengths = lengths(ang_freqs, sample_params, power_in, pulse_duration)
    dashes = "-" *90
    print("Coefficiets at central wavelengths")
    print("#"*90)
    print("optimal pressure: " + str(opt_pressure) + " bar")
    print(dashes)
    print("gammas: " + str(gammas) + " 1/(m*W)")
    print("L_NL: " + str(charasteritic_lengths[0]) + " m")
    print(dashes)
    print("alhpa: " + str(alphas))
    print(dashes)
    print("beta: " + str([beta0_p,beta0_s,beta0_i]) + " 1/m")
    print("phase mismatch: " + str(db) + " 1/m")
    print(dashes)
    print("beta_1: " + str([beta1_p*1e12,beta1_s*1e12,beta1_i*1e12]) + " ps/m")
    print("group velocity mismatch(sp,ip): " + str([d_sp*1e15, d_ip*1e15]) + " fs/m")
    print("L_W(sp,ip): " + str(charasteritic_lengths[1]) + " m")
    print(dashes)
    print("beta_2: " + str([beta2_p*1e24,beta2_s*1e24,beta2_i*1e24]) + " ps^2/m")
    print("L_D: " + str(charasteritic_lengths[2]) + " m")
    print(dashes)
    print("beta_3: " + str([beta3_p*1e36,beta3_s*1e36,beta3_i*1e36]) + " ps^3/m")
    print("L_T: " + str(charasteritic_lengths[3]) + " m")

    #SOLVER & PLOTS
    z, A = propagator(length, ang_freqs, w_w0, fields_in, params, time_step)

    fig, ax = plt.subplots(2,3, subplot_kw=dict(projection='3d'))
    fig.set_size_inches(16, 10)

    # fig_frog, ax_frog = plt.subplots(1,3)
    # fig_frog.set_size_inches(16, 5)
    # frog_plots_p = [ ]
    # frog_plots_s = [ ]
    # frog_plots_i = [ ]

    index = 0
    z_plotted = []
    while index < len(z)-1:
        ax[0,0].plot(t*1e15, np.abs(A[index,0])**2*pulse_duration*1e6, z[index]*100, zdir='y', color='blue')
        ax[0,1].plot(t*1e15, np.abs(A[index,1])**2*pulse_duration*1e6, z[index]*100, zdir='y', color='red')
        ax[0,2].plot(t*1e15, np.abs(A[index,2])**2*pulse_duration*1e6, z[index]*100, zdir='y', color='purple')
        ax[1,0].plot((l_l0+2*pi*c/wp)*1e9, np.fft.fftshift(np.abs( np.fft.fft(A[index,0]) ))*1e-5, z[index]*100, zdir='y', color='blue')
        ax[1,1].plot((l_l0+2*pi*c/ws)*1e9, np.fft.fftshift(np.abs( np.fft.fft(A[index,1]) ))*1e-5, z[index]*100, zdir='y', color='red')
        ax[1,2].plot((l_l0+2*pi*c/wi)*1e9, np.fft.fftshift(np.abs( np.fft.fft(A[index,2]) ))*1e-5, z[index]*100, zdir='y', color='purple')

        # frog_p = frogtr(A[index+1,0], A[index+1,0])
        # im_p = ax_frog[0].imshow(np.abs(frog_p)**2, aspect='auto', extent=(t[0]*1e15, t[-1]*1e15, (l_l0[0]+2*pi*c/wp)*1e9 , (l_l0[-1]+2*pi*c/wp)*1e9))
        # frog_plots_p.append([im_p])
        #
        # frog_s = frogtr(A[index+1,1], A[index+1,1])
        # im_s = ax_frog[1].imshow(np.abs(frog_s)**2, aspect='auto', extent=(t[0]*1e15, t[-1]*1e15, (l_l0[0]+2*pi*c/ws)*1e9, (l_l0[-1]+2*pi*c/ws)*1e9))
        # frog_plots_s.append([im_s])
        #
        # frog_i = frogtr(A[index+1,2], A[index+1,2])
        # im_i = ax_frog[2].imshow(np.abs(frog_i)**2, aspect='auto', extent=(t[0]*1e15, t[-1]*1e15, (l_l0[0]+2*pi*c/wi)*1e9, (l_l0[-1]+2*pi*c/wi)*1e9))
        # frog_plots_i.append([im_i])
        #
        # z_plotted.append(z[index])

        index += len(z)//25 #number on bottom gives estimate of how many lines will be plotted for visualization of pulse shape over distance
    for j in [0,1,2]:
        ax[0,j].set_xlabel('Time (fs)')
        ax[0,j].set_ylabel('Distance (cm)')
        ax[0,j].set_zlabel('Energy (uJ)')
        ax[1,j].set_xlabel('Wavelength (nm)')
        ax[1,j].set_ylabel('Distance (cm)')
        ax[1,j].set_zlabel('Amplitude (a.u.)')
        # ax_frog[j].set_title('FROG Trace')
        # ax_frog[j].set_xlabel('Time [fs]')
        # ax_frog[j].set_ylabel('Wavelength (nm)')
        # plt.colorbar(im_p, ax=ax_frog[j])

    # frog_anim_p = animation.ArtistAnimation(fig=fig_frog, artists=frog_plots_p)
    # frog_anim_s = animation.ArtistAnimation(fig=fig_frog, artists=frog_plots_s)
    # frog_anim_i = animation.ArtistAnimation(fig=fig_frog, artists=frog_plots_i)
    # frog_anim_p.save(filename="/Users/davidgarcia/Desktop/frog_UV.gif", writer="pillow")
    # frog_anim_s.save(filename="/Users/davidgarcia/Desktop/frog_UV.gif", writer="pillow")
    # frog_anim_i.save(filename="/Users/davidgarcia/Desktop/frog_UV.gif", writer="pillow")
    plt.show()

main()

'''
References
1. Yongzhong Li et al 2006 J. Opt. A: Pure Appl. Opt. 8 689
2. H. J. Lehmeier, W. Leupacher, and A. Penzkofer, Opt. Commun. 56, 67 (1985).
3. A. Börzsönyi, Z. Heiner, M. P. Kalashnikov, A. P. Kovács, and K. Osvay, “Dispersion measurement of inert gases and gas mixtures at 800 nm,” Appl. Opt. 47, 4856–4863 (2008).
4. E. A. J. Marcatili and R. A. Schmeltzer, "Hollow metallic and dielectric waveguides for long distance optical transmission and lasers," in The Bell System Technical Journal, vol. 43, no. 4, pp. 1783-1809, July 1964, doi: 10.1002/j.1538-7305.1964.tb04108.x.
5. I. H. Malitson, "Interspecimen Comparison of the Refractive Index of Fused Silica*,†," J. Opt. Soc. Am. 55, 1205-1209 (1965)
6. John C. Travers, Wonkeun Chang, Johannes Nold, Nicolas Y. Joly, and Philip St. J. Russell, "Ultrafast nonlinear optics in gas-filled hollow-core photonic crystal fibers [Invited]," J. Opt. Soc. Am. B 28, A11-A26 (2011)
7. S. Balac and F. Mahe´, Embedded Runge-Kutta scheme for step-size control in the interaction picture method, Computer Physics Communications, 184 (2013), pp. 1211–1219.
8. J P Siqueira et al 2016 J. Phys. B: At. Mol. Opt. Phys. 49 195601
'''
