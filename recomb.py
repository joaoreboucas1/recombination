"""
    A study of recombination
    Author: João Rebouças, February 2026
    The goal of this script is to reproduce Figure 4.1 from the book
    'The Cosmic Microwave Background', by Ruth Durrer
"""

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

c_in_km_s = 277_792.458
apery = 1.2020569031
eta = 1e-9
me = 0.510998e6 # eV
B_H = 13.6      # eV
kelvin_to_ev = 8.61732e-5
T_CMB = 2.7255*kelvin_to_ev
chbar = 197_326_980e-15 * 3.24e-23 # ev*Mpc
Y_he = 0.24
sigma_T = 6.6524e-25 * (3.24e-25)**2 # Mpc^2

def get_saha_x_e(a):
    T = T_CMB/a
    lhs = 2*apery*eta/pi**2 * (2*pi*T/me)**1.5 * np.exp(B_H/T)
    x_e = (np.sqrt(1 + 4*lhs) - 1)/(2*lhs)
    T_g = 0.24 # eV
    # NOTE: implementing manual freeze-out
    # Index of first value of T that is below T_g
    index = np.argmax(T <= T_g)
    x_e[T <= T_g] = x_e[index]
    return x_e

def phot_number_density(a):
    T = T_CMB/a
    return 2*apery*T**3/pi**2/(chbar)**3 # 1/Mpc^3

def get_collision_rate(a):
    x_e = get_saha_x_e(a)
    n_gamma = phot_number_density(a)
    n_e = x_e*(1-Y_he)*eta*n_gamma
    kappa_prime = a*n_e*sigma_T # 1/Mpc
    return x_e, n_e, kappa_prime

# To integrate kappa_prime with respect to conformal time, we need cosmology
Omega_m = 0.3
Omega_r = 2.5e-5
Omega_de = 1 - Omega_m - Omega_r
H0 = 70/c_in_km_s
def H_curly(a):
    return H0*a*np.sqrt(Omega_r*a*-4 + Omega_m*a**-3 + Omega_de) # 1/Mpc

def integrate_collision_rate(a, kappa_prime):
    kappa = np.zeros_like(kappa_prime)
    H = H_curly(a)
    log_a = np.log(a)
    for i in range(len(kappa_prime)):
        kappa[i] = np.trapezoid(kappa_prime[i+1:]/H[i+1:], log_a[i+1:])
    return kappa

a = np.logspace(-4, 0, 1000)
z = 1/a - 1
x_e, n_e, kappa_prime = get_collision_rate(a)
kappa = integrate_collision_rate(a, kappa_prime)
visibility = kappa_prime*np.exp(-kappa)

fig, axs = plt.subplots(2, 2, figsize=(11, 11))
fig.suptitle("Recombination quantities")

axs[0,0].set_title("Free electron fraction")
axs[0,0].loglog(a, x_e)
axs[0,0].set_xlim([a[0], a[-1]])
axs[0,0].set_ylim([1e-4, 2])
axs[0,0].set_xlabel("$a$")
axs[0,0].set_ylabel("$x_e$")

axs[0,1].set_title("Collision rate")
axs[0,1].loglog(a, kappa_prime)
axs[0,1].set_xlim([a[0], a[-1]])
axs[0,1].set_xlabel("$a$")
axs[0,1].set_ylabel("$an_e\\sigma_T \\, [1/\\mathrm{Mpc}^3]$")

axs[1,0].set_title("Optical depth")
axs[1,0].plot(z, kappa)
axs[1,0].set_xlim([450, 2000])
axs[1,0].set_ylim([0, 60])
axs[1,0].set_xlabel("$z$")
axs[1,0].set_ylabel("$\\kappa$")

axs[1,1].set_title("Visibility function")
axs[1,1].plot(z, visibility/H0)
axs[1,1].set_xlim([450, 2000])
axs[1,1].set_xlabel("$z$")
axs[1,1].set_ylabel("$g(z)$")

plt.savefig("Ruth_Durrer_Reproduction.pdf", bbox_inches="tight")