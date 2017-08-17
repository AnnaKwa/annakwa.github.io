import numpy as np
import matplotlib.pyplot as plt
from gatspy import periodic
from IPython.display import clear_output
from scipy.optimize import curve_fit
from scipy.stats import chisquare

from matplotlib import rcParams
rcParams.update({'font.size': 18})


def sine_model(x, amp, ph, offset):
    # fixed freq b/c fitting phase folded data: x ranges from 0-1
    return amp*np.sin(x * 2.*np.pi + ph)+offset

def chi2dof(x, y_obs, y_model, sigma):
    sum=0.
    dof=len(x)-1
    for i in range(len(x)):
        sum+= (y_obs[i]-y_model[i])**2/y_model[i]**2
    return sum/dof

class Star:
    '''
    data from Lick-Carnegie Exoplanet Survey (https://home.dtm.ciw.edu/ebps/data/)
    assumes file name formatted as [star_ID]_KECK.vel_dependent_cross_section
    columns: Julian date, velocity [m/s], sig_vel [m/s], S_value, Halpha, median photons/pix, exposure time [s]
    '''
    def __init__(self, star_ID, data_dir='data/Keck_RVs'):
        #import data
        self.t, self.v, self.sig_v, sval, Halpha, phot_pix, exp_t = np.loadtxt(data_dir+'/'+star_ID+'_KECK.vels', unpack=True)
        self.ID = star_ID
        self.N = len(self.t)

        # subtract off starting observation date so that first data point is t=0
        start_t = np.min(self.t)
        self.t = self.t - start_t

    def find_period(self, num_best_fits=3, min_period=0.07, max_period=10, print_output=False):
        self.min_period, self.max_period = min_period, max_period
        self.model = periodic.LombScargleFast(fit_period=True,silence_warnings=True)
        self.model.optimizer.period_range = (self.min_period, self.max_period)
        self.model.fit(self.t, self.v, self.sig_v)

        # find and save best fitting periods for data
        self.best_periods = self.model.find_best_periods(num_best_fits, return_scores=True)
        clear_output()
        if print_output==True:
            print('Best-fitting periods:')
            for k in range(num_best_fits):
                print('P=',self.best_periods[0][k], ' days , power=',self.best_periods[1][k])

    def fit_to_sine(self, fitNum=0):
        # fitNum = 0 : fit to period value with highest score
        # fitNum = 1: fit to second highest score, etc.
        # returns chi2 stat, best-fitting amplitude A, phase c, offset b for sine fit to phase-folded data
        N_obs = len(self.t)
        phase = []
        P = self.best_periods[0][fitNum]
        for i in range(N_obs):
            phase_folded_t = self.t[i]%P
            phase.append(phase_folded_t/P)
        guess_amplitude = 0.5*( abs(np.max(self.v)) + abs(np.min(self.v)) )
        guess_phase = 0.3*np.pi
        guess_offset = 0.
        initial_guesses = [guess_amplitude, guess_phase, guess_offset]
        fit = curve_fit(sine_model, phase, self.v, p0=initial_guesses)
        A, c, b = fit[0][0], fit[0][1], fit[0][2]
        chi2 = chi2dof(self.t, self.v, sine_model(np.array(phase),A,c,b), self.sig_v)
        return (A, c, b, chi2)

    def plot_raw_RVs(self, save=False):
        # plot RV curve with x-axis in units of observation date (starting at t=0 days)
        fig= plt.figure(figsize=(15,5))
        ax=fig.add_subplot(111)
        ax.errorbar(self.t, self.v, yerr=self.sig_v, fmt='o')
        ax.set_xlabel('Observation date [Days]')
        ax.set_ylabel('Radial velocity [m/s]')
        ax.set_title(self.ID, fontsize=20)

        if save==True:
            plt.savefig('RVcurve_raw_'+self.ID+'.png')
        plt.show()

    def find_periodogram(self, Nperiods=5000, plot=False, save=False):
        # plot power at various periods
        p_arr = np.linspace(self.min_period, self.max_period, Nperiods)
        power = self.model.score(p_arr)
        avg_power = np.average(power)

        if plot==True:
            fig= plt.figure(figsize=(6,5))
            ax=fig.add_subplot(111)
            ax.plot(p_arr, power)
            ax.set_xlabel('Period [days]')
            ax.set_ylabel('Power')
            ax.set_xlim([self.min_period, self.max_period])
            ax.set_ylim([0.,1.])

            if save==True:
                plt.savefig('periodogram'+_self.ID+'.png')
            plt.show()


    def plot_phased_RVs(self, num_plots=1, save=False):
        period_arr, phase_arr = [],[]
        for j in range(num_plots):
            P = self.best_periods[0][j]
            period_arr.append(P)
            phase_temp_arr = []
            for i in range(self.N):
                phase_folded_t = self.t[i] % P
                phase_temp_arr.append(phase_folded_t/P)
            phase_arr.append(phase_temp_arr)

        fig = plt.figure(figsize=(5*num_plots,4))
        x_arr = np.linspace(0,1,200)

        for p in range(num_plots):
            ax=fig.add_subplot(1,num_plots,p+1)
            P = period_arr[p]
            sinefit = self.fit_to_sine(fitNum=p)
            A, c, b = sinefit[0], sinefit[1], sinefit[2]
            fit_curve = sine_model(np.array(x_arr), A, c, b)
            ax.plot(x_arr, fit_curve, '--', linewidth=2.6, color='tomato')
            ax.errorbar(phase_arr[p], self.v, yerr=self.sig_v, fmt='o', markersize=6)
            ax.annotate('P = '+"%.4f" % P+' days',
                        xy=(0.97,0.05),
                        backgroundcolor='white',
                        xycoords='axes fraction',
                        horizontalalignment='right',
                        size=16)
            if p==0:
                ax.set_ylabel('Radial velocity [m/s]')
            if num_plots==2 or num_plots==1:
                ax.set_xlabel('Phase')
                if num_plots==1:
                    ax.set_title(self.ID, fontsize=20)
                if num_plots==2 and p==0:
                    ax.set_title(self.ID, fontsize=20)
            else:
                if p==1:
                    ax.set_xlabel('Phase')
                    ax.set_title(self.ID, fontsize=20)
        if save==True:
            plt.savefig('RVcurve_phased_'+self.ID+'.png')
        plt.show()
