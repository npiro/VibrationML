from detect_peaks import detect_peaks
import numpy as np
from matplotlib.pyplot import plot
from sklearn import svm

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
        th = 0.05*(M-m)
        ind = detect_peaks(data,mph=mph,threshold=th,mpd=8)
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

                
    def getFundamentalByPeakDetect(self):
        if self.have_data:
            if not self.peaks_found:
                peakindeces = self.findPeaks(self.amp)                
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
        peaks = np.array([self.findPeaks(a)[:peaksPerSpectrum] for a in amps])
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
        
        
    def SVManom(self):
        self.clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        
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