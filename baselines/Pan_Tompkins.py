import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
 
class ECGProcessor:

    def __init__(self, fs=200):

        self.fs = fs

        self.ms_to_samples = lambda x: int(x * self.fs / 1000)  # Convert ms to samples

    def butter_bandpass(self, lowcut=5, highcut=15, order=2):

        """Create butterworth bandpass filter"""

        nyq = 0.5 * self.fs

        low = lowcut / nyq

        high = highcut / nyq

        b, a = butter(order, [low, high], btype='band')

        return b, a

    def filter_signal(self, signal):

        """Apply bandpass filter to remove noise"""

        b, a = self.butter_bandpass()

        return filtfilt(b, a, signal)

    def derivative_filter(self, signal):

        """Apply derivative filter to highlight QRS slopes"""

        # More sophisticated derivative filter

        derivative = np.zeros_like(signal)

        for i in range(2, len(signal)-2):

            derivative[i] = (-signal[i-2] - 2*signal[i-1] + 2*signal[i+1] + signal[i+2])/(8/self.fs)

        return derivative

    def adaptive_threshold(self, signal, peaks, window_size=150):

        """Calculate adaptive threshold based on local signal characteristics"""

        threshold = np.zeros_like(signal)

        for i in range(len(signal)):

            start = max(0, i - window_size)

            end = min(len(signal), i + window_size)

            local_mean = np.mean(signal[start:end])

            local_std = np.std(signal[start:end])

            threshold[i] = local_mean + 2 * local_std

        return threshold

    def find_qrs_peaks(self, signal, min_distance=200):

        """Detect QRS complexes using adaptive thresholding"""

        # Initial peak detection

        filtered = self.filter_signal(signal)

        derivative = self.derivative_filter(filtered)

        squared = derivative ** 2

        # Moving window integration

        window_size = self.ms_to_samples(160)  # 150ms window

        mwi = np.convolve(squared, np.ones(window_size)/window_size, 'same')

        # Adaptive thresholding

        peaks, _ = find_peaks(mwi, distance=self.ms_to_samples(min_distance))

        threshold = self.adaptive_threshold(mwi, peaks, window_size=100)

        # Refine peaks using threshold

        qrs_peaks = []

        for peak in peaks:

            if mwi[peak] > threshold[peak]:

                qrs_peaks.append(peak)

        return np.array(qrs_peaks), filtered, mwi, threshold

    def find_pqrst(self, signal, qrs_peaks):

        """Detect P, Q, R, S, and T waves around QRS complexes"""

        pqrst_points = []

        # Search windows (in ms)

        p_window = (-200, -50)  # Search for P wave

        q_window = (-50, 0)     # Search for Q wave

        s_window = (0, 50)      # Search for S wave

        t_window = (0, 450)    # Search for T wave with respect to your S index, not the R.

        filtered_signal = self.filter_signal(signal)

        for r_peak in qrs_peaks:

            wave_points = {'R': r_peak}
 
            # Find P wave (maximum before Q)

            p_start = max(0, r_peak + self.ms_to_samples(p_window[0]))

            p_end = max(0, r_peak + self.ms_to_samples(p_window[1]))

            if p_start < p_end:

                p_idx = np.argmax(filtered_signal[p_start:p_end]) + p_start

                wave_points['P'] = p_idx
 
            # Find Q wave (minimum before R)

            q_start = max(0, r_peak + self.ms_to_samples(q_window[0]))

            q_end = r_peak + self.ms_to_samples(q_window[1])

            if q_start < q_end:

                q_idx = np.argmin(filtered_signal[q_start:q_end]) + q_start

                wave_points['Q'] = q_idx

            # Find S wave (minimum after R)

            s_start = r_peak + self.ms_to_samples(s_window[0])

            s_end = min(len(signal), r_peak + self.ms_to_samples(s_window[1]))

            if s_start < s_end:

                s_idx = np.argmin(filtered_signal[s_start:s_end]) + s_start

                wave_points['S'] = s_idx

            # Find T wave (maximum after S)

            # t_start = r_peak + self.ms_to_samples(t_window[0])

            t_start = s_idx + self.ms_to_samples(t_window[0])

            t_end = min(len(signal), r_peak + self.ms_to_samples(t_window[1]))

            if t_start < t_end:

                t_idx = np.argmax(filtered_signal[t_start:t_end]) + t_start

                wave_points['T'] = t_idx

            pqrst_points.append(wave_points)

        return pqrst_points

    def plot_results(self, signal, pqrst_points, filtered=None):

        """Plot ECG with detected PQRST points"""

        t = np.arange(len(signal)) / self.fs

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Plot original signal

        ax1.plot(t, signal, label='Original')

        ax1.set_title('Original ECG Signal')

        # Plot filtered signal if available

        if filtered is not None:

            ax2.plot(t, filtered, label='Filtered')

            ax2.set_title('Filtered ECG Signal with PQRST Detection')

        # Plot PQRST points

        colors = {'P': 'g', 'Q': 'm', 'R': 'r', 'S': 'b', 'T': 'c'}

        legend_added = set()  # Keep track of which waves we've added to legend

        for points in pqrst_points:

            for wave, idx in points.items():

                if filtered is not None:

                    label = f'{wave} wave' if wave not in legend_added else ''

                    ax2.plot(t[idx], filtered[idx], 'o', color=colors[wave], label=label)

                    ax1.plot(t[idx], signal[idx], 'o', color=colors[wave], label=label)

                    legend_added.add(wave)

        ax1.legend()

        ax2.legend()

        plt.tight_layout()

        return fig, (ax1, ax2)