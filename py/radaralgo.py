# compute range and velocity using a stream of IQ sequences
# yves.piguet@csem.ch

import numpy as np
import math

class RadarAlgo:

    c = 3e8

    def __init__(self, Tc, B, freq_min, Ts, d=1e-2, forgetting_factor=0.9):
        self.Tc = Tc
        self.B = B
        self.freq_min = freq_min
        self.Ts = Ts
        self.d = d

        self.forgetting_factor = forgetting_factor

        self.reset()
    
    def reset(self):
        self.t = 0
        self.phase = 0
        self.velocity = 0

    def freq(self, R):
        """
        Calculate the baseband range-dependent frequency.

        Parameter:
          R: range
        Return:
          frequency
        """
        f = self.freq_min + 2 * self.B * R / (self.c * self.Tc)
        return f

    @staticmethod
    def estimate_freq_peak(vec):
        """
        Calculate the FFT of the sample vector and find the index and phase of its maximum

        Parameter:
          vec: vector of samples
        Return:
          index, phase
        """
        f = np.fft.fft(np.array(vec))
        i_max = np.argmax(np.abs(f))
        return i_max, np.angle(f[i_max])

    def estimate_range(self, vec, direction):
        """
        Estimate the range of the target with the largest echo using a sample vector

        Parameters:
          vec: vector of samples
          direction: sweep direction (True=up, False=down) (ignored)
        
        Return:
          range, phase
        """
        i_max, phase = self.estimate_freq_peak(vec)
        f_IF = i_max / (self.Ts * len(vec))
        R = self.c * self.Tc * f_IF / (2 * self.B)
        return R, phase

    def estimate_range_velocity(self, vec, direction):
        """
        Estimate the range and velocity of the targer with the largest echo using
        a sample vector. The velocity uses the phase estimated by the previous call
        and filtered with a first-order low-pass filter.

        Parameters:
          vec: vector of samples
          direction: sweep direction (True=up, False=down) (ignored)
        
        Return:
          range, velocity
        """
        R, phase = self.estimate_range(vec, direction)
        lambda_max = self.c / self.freq_min
        velocity = lambda_max * (phase - self.phase) / (4 * math.pi * self.Tc)
        self.phase = phase
        self.velocity = (self.forgetting_factor * self.velocity
                         + (1 - self.forgetting_factor) * velocity)
        return R, self.velocity

    def estimate_range_angle_velocity(self, vec1, vec2, direction):
        """
        Estimate the range, the angle and the velocity of the target with the
        largest echo using two sample vectors, one per RX antenna

        Parameters:
          vec1: vector of samples for the first RX antenna
          vec2: vector of samples for the second RX antenna
          direction: sweep direction (True=up, False=down) (ignored)
        
        Return:
          range, angle, velocity
        """
        R1, phase1 = self.estimate_range(vec1, direction)
        R2, phase2 = self.estimate_range(vec2, direction)
        # print("phase1", np.degrees(phase1), "phase2", np.degrees(phase2))
        R = 0.5 * (R1 + R2)
        # angle
        lambd = self.c / self.freq(R)
        # phase difference in [-pi, pi)
        delta_phase = (phase2 - phase1 + math.pi) % (2 * math.pi) - math.pi
        # angle based on phase difference
        alpha = math.asin(lambd * delta_phase / (2 * math.pi * self.d))
        # velocity based on vec1
        velocity1 = lambd * (phase1 - self.phase) / (4 * math.pi * self.Tc)
        self.phase = phase1
        self.velocity = (self.forgetting_factor * self.velocity
                         + (1 - self.forgetting_factor) * velocity1)
        return R, alpha, velocity1

    def range_doppler_fft(self, array):
        """
        Apply the range-doppler FFT to an array of samples. Each row is a sample
        vector for a single upward sweep, and successive rows are for successive
        sawtooth sweeps. 

        Parameter:
          array: array of samples
        
        Return:
          r_d_fft: array of complex values; r_d_fft[i,j] has range=scale_fun(i,j)[0][j]
          and velocity=scale_fun(i,j)[1][i], and obstacles have a large magnitude
          scale_fun: function to calculate range and velocity for item i,j 
        """
        r_d_fft = np.fft.fft2(array)

        # scale
        n_num_sweeps, n_samples_per_sweep = array.shape
        f_IF_vec = [i / (self.Ts * n_samples_per_sweep) for i in range(n_samples_per_sweep)]
        def scale_fun(i, j):
            R = self.c * self.Tc * f_IF_vec[j] / (2 * self.B)
            lambd = self.c / self.freq(R)  # lambd = self.c / self.freq_min ???
            velocity = lambd * (i if i < n_num_sweeps // 2 else i - n_num_sweeps) / (2 * self.Tc * n_num_sweeps)
            return R, velocity

        return r_d_fft, scale_fun

    def synthetic_samples(self, R, velocity=0, alpha=0, vec_len=1024, noise=0):
        """
        Generate a vector of complex samples.

        Parameters:
          R: range
          velocity: velocity
          alpha: target angle in radians
          vec_len: vector length (default: 1024)
          noise: standard deviation of gaussian noise on I and Q (default: 0)
        """
        f_IF = 2 * self.B * R / (self.c * self.Tc)
        i_max = int(round(f_IF * self.Ts * vec_len % vec_len))
        lambd = self.c / self.freq_min
        self.phase += 4 * math.pi * self.Tc * velocity / lambd
        phase = self.phase + 2 * math.pi * math.sin(alpha) * self.d / lambd
        fft = np.zeros((vec_len), dtype=complex)
        fft[i_max] = np.exp(1j * phase)
        if noise != 0:
            fft += noise * (np.random.normal(size=fft.shape)
                             + 1j * np.random.normal(size=fft.shape))
        vec = np.fft.ifft(fft)
        return vec

    def synthetic_samples_direct(self, R, velocity=0, noise=0, upward_chirp=True):
        """
        Generate a vector of complex samples with direct equation for mixer output,
        without IFFT. Vector length is Tc / Ts.

        Parameters:
          R: range
          velocity: velocity
          noise: standard deviation of gaussian noise on I and Q (default: 0)
        """
        vec_len = int(self.Tc // self.Ts)
        t = self.t + np.arange(vec_len) * self.Ts
        self.t += self.Tc
        S = self.B / self.Tc
        tau = 2 * R / self.c
        if upward_chirp:
          phase_der = 4 * math.pi / self.c * self.freq_min * velocity
          vec = np.exp(2j * math.pi * (S * tau * t - 0.5 * S * tau ** 2 + self.freq_min * tau) + 1j * self.phase)
        else:
          phase_der = -4 * math.pi / self.c * (self.freq_min + self.B) * velocity
          vec = np.exp(2j * math.pi * (S * tau * t - 0.5 * S * tau ** 2 - (self.freq_min + self.B) * tau) + 1j * self.phase)
        self.phase += phase_der * self.Tc
        return vec
