from detect_peaks import detect_peaks
import numpy as np
from matplotlib.pyplot import plot
from sklearn import svm
import peakutils


#%% FFT
def CalculateFramesFFT(frames, rate = 8000, chunksize = 2048, avgwind = 6, plot = False):
    '''
    Calculate FFT for all frames.
    frames: np array of time series signals
    rate: sampling rate. Default: 8000
    chunksize: size of each chunk of data. Default: 2048
    avgwind: 6
    plot: If true, plots the results
    '''
    freq_vect = np.fft.rfftfreq(chunksize, 1./rate)
    fft_frames = np.array([np.fft.rfft(fr) for fr in frames])
    fft_frames /= np.abs(fft_frames).max()
    fft_frames = np.abs(fft_frames)
    #subframes = fft_frames[0:avgwind,:]
    fft_frames = np.array([np.mean(fft_frames[i:i+avgwind,:],0) for i in range(0,len(fft_frames),avgwind)])

    if plot:
        plt.plot(freq_vect,fft_frames[0])
        plt.hold(True)
        for i in range(1,len(fft_frames)):
            plt.plot(freq_vect, fft_frames[i])
        plt.hold(False)
        ax = plt.gca()
        ax.set_yscale('log')
    return (freq_vect, fft_frames)
#%%
def GetFeaturesFromFrames(freq_vect, fft_frames, num_features = 6, sp = None, min_freq = 90):
    '''
    from the fft of all frames, computes feature vectors
    num_features: number of features to compute
    sp: SoundProcessor object. If not specified, will generate a new one
    Returns: (num_samples, num_features) array of features
    '''
    if sp is None:
        sp = SpectrumProcessor()
    
    ok = (freq_vect > min_freq)
    freq_vect = freq_vect[ok]
    fft_frames[:,ok]
    peak_inds = sp.getPeaksFromSpectrogram(freq_vect,fft_frames,num_features)
    maxinds = [np.argmax(a[peak_inds[i][0]]) for i, a in enumerate(fft_frames)]

    funds = [freq_vect[peak_ind[0][maxinds[i]]] for i, peak_ind in enumerate( peak_inds)]
    xpeaks = [freq_vect[peak_ind[0]]/funds[i] for i, peak_ind in enumerate(peak_inds)]
    ypeaks = [np.array(fft_frames[i][peak_ind[0]]) for i, peak_ind in enumerate(peak_inds)]
    
    sortinds = [np.argsort(xpeak) for xpeak in xpeaks]    
    xpeaks_sorted = [xpeak[sortind] for xpeak, sortind in zip(xpeaks,sortinds)]
    ypeaks_sorted = [ypeak[sortind] for ypeak, sortind in zip(ypeaks,sortinds)]      
                     
    X = [np.hstack((xpeaks_sorted, ypeaks_sorted))][0]
         
    return X

class SpectrumProcessor(object):
    def __init__(self,freq = None, amp = None):
        self.peaks_found = False
        self.have_data = False
        self.have_freq = False
        self.have_amp = False
        
        if amp is not None:
            self.amp = amp
            self.have_amp = True
            self.peaks_indeces = self.findPeaks(amp)
            self.peaks_found = True
        if freq is not None:
            self.freq = freq
            self.have_freq = True
            
            
    # Detect peaks in spectrum
    def findPeaks(self,data=None):
        if data is None:
            data = self.amp
        M = np.max(data)
        m = np.min(data)
        mph = m + 0.07*(M-m)
        #mph = 0.07*(M-m)
        th = 0.05*(M-m)
        ind = detect_peaks(data,mph=mph,threshold=th,mpd=8)
        peak_values = data[ind]
        self.peaks_found = True
        self.peak_indeces = ind
        self.peak_values = peak_values
        self.peak_freqs = self.freq[ind]
        return ind
    
        
    def findPeaksPU(self,data=None):
        if data is None:
            data = self.amp
        M = np.max(data)
        m = np.min(data)
        mph = m + 0.07*(M-m)
        #mph = 0.07*(M-m)
        th = 0.05*(M-m)
        ind = peakutils.indexes(data, thres=0.02/max(data), min_dist=8)
        peak_values = data[ind]
        self.peaks_found = True
        self.peak_indeces = ind
        self.peak_values = peak_values
        self.peak_freqs = self.freq[ind]
        return ind
        
    def setData(self,freq,amp):
        self.freq = freq
        self.amp = amp
        self.have_data = True
        self.have_freq = True
        self.have_amp = True
        self.peaks_found = False
        self.peak_indeces = []
        self.peak_values = []
        self.peak_freqs = []
        
    def setFreq(self,freq):
        self.freq = freq
        self.have_freq = True
        if self.have_amp:
            self.have_data = True
        self.peaks_found = False
        self.peak_indeces = []
        self.peak_values = []
        self.peak_freqs = []
        
    def setAmp(self,amp):
        self.amp = amp
        self.have_amp = True
        if self.have_freq:
            self.have_data = True
        self.peaks_found = False
        self.peak_indeces = []
        self.peak_values = []
        self.peak_freqs = []
    
    def getFundamentalByHPS(self):
        if self.have_data:
            try:
                a1 = self.amp
                a2 = a1[::2]
                a3 = a1[::3]
                aP = a3*a2[:len(a3)]*a1[:len(a3)]
                MI = max(enumerate(aP), key=lambda p: p[1])[0]
                return self.freq[MI], aP, MI
                #ind = self.findPeaks(aP)
                #peakAmps = aP[ind]
                #peakFreqs = self.freq[ind]
                #maxPeakInd = max(enumerate(peakAmps), key=operator.itemgetter(1))
                
                #return self.freq[ind]
                
            except Exception as e:
                print('Error: {0}'.format(e))

    def getFundamentalByMaximum(self):
        if self.have_data:
            if not self.peaks_found:
                peakindeces = self.findPeaks(self.amp)                
            else:
                peakindeces = self.peak_indeces   
                
    def getFundamentalByPeakDetect(self):
        if self.have_data:
            if not self.peaks_found:
                peakindeces = self.findPeaks(self.amp) 
                #peakindeces = self.findPeaksPU(self.amp)                

            else:
                peakindeces = self.peak_indeces                
                
            try:
                peak_freqs = self.peak_freqs
                peakvals = self.peak_values
                if len(peak_freqs) > 1:
                    freqdist = peak_freqs[1:]-peak_freqs[:-1]
                elif len(peak_freqs) == 1:
                    freqdist = peak_freqs
                else:
                    return -1
                    
                hist, edges = np.histogram(freqdist, density = True)
                sortindeces = np.argsort(hist)[::-1]
                centers = 0.5*(edges[1:]+edges[:-1])
                if len(peak_freqs) > 1:
                    return centers[sortindeces[0]], hist[sortindeces], hist, edges
                else:
                    return freqdist[0], hist[sortindeces], hist, edges
            except Exception as e:
                print('Error: {0}'.format(e))
                print(hist)
                print(sortindeces)

    def getPeaksFromSpectrogram(self,freq=None,amps=None,peaksPerSpectrum = 10):
        if amps is None or freq is None:
            return None
        self.setFreq(freq)   
        peaks = []
        for a in amps:
            pi = self.findPeaksPU(a)
            sortedind = np.argsort(a[pi])[::-1]
            
            peaks.append(np.array([pi[sortedind[:peaksPerSpectrum]]]))
        return peaks

class MachineLearning:
    

    def __init__(self,ML_model = 'SVM anomaly'):

        self.ML_model = ML_model
        functions = {'SVM anomaly': self.SVManom,'SVC': self.SVC, 'NN': self.NN}

        func = functions[ML_model]
        func()
    
    def ExtractFeatures(self,freq,amps,num_features = 10):
        sp = SpectrumProcessor()
        peak_inds = sp.getPeaksFromSpectrogram(freq,amps,num_features)
        
        peak_freq = freq[peak_inds]
        peak_amps = amps[np.arange(amps.shape[0])[:, None], peak_inds]
        sorted_peak_inds = np.argsort(peak_amps,axis=1) # Sort matrix rows (getting sorting index)
        sorted_peak_amps = peak_amps[np.arange(peak_amps.shape[0])[:,np.newaxis],sorted_peak_inds]    # apply to amps
        sorted_peak_freqs = peak_freq[np.arange(peak_freq.shape[0])[:,np.newaxis],sorted_peak_inds]
        #sorted_peak_freq = peak_freq[sorted_peak_inds[:num_features]] 
        self.features = sorted_peak_amps
        
        
    def SVManom(self, nu=0.1, kernel="rbf", gamma=0.1):
        self.clf = svm.OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        
    def SVC(self):
        self.clf = svm.SVC()
        clf.fit(X, y) 
    
    def NN(self):
        pass
    
    def learnSpectrum(self,cla=None):
        if not self.peaks_found:
            self.findPeaks()
        peak_freqs = self.peak_freqs
        peak_vals = self.peak_values
    
    def validateSpectrum(self,cla=None):
        pass
    
    def predictSpectrum(self):
        pass