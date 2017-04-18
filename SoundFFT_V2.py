import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtGui, QtCore, uic, QtWidgets
from MplFigure import MplFigure
from MicrophoneRecorder import MicrophoneRecorder
from SpectrumProcessor import SpectrumProcessor, MachineLearning, GetFeaturesFromFrames, CalculateFramesFFT
#import seaborn
import traceback

Ui_MainWindow, QMainWindow = uic.loadUiType("VibrationGUI_V2.ui") 

class VibrationAnalysisMainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    TRAINING_SIZE_COL = 0
    VALIDATION_SIZE_COL = 1
    LABEL_COL = 2
    
    def __init__(self):
        
        super(VibrationAnalysisMainWindow, self).__init__()
        self.setupUi(self)
        
        # customize the UI
        self.initUI()
        
        # init class data
        self.initData()       
        
        # connect slots
        self.connectSlots()
        
        # init MPL widget
        self.initMplWidget()
        
    def initUI(self):
         
        # mpl figure
        self.main_figure = MplFigure(self)
        self.PlotLayout.addWidget(self.main_figure.toolbar)
        self.PlotLayout.addWidget(self.main_figure.canvas)

        
        #self.setGeometry(300, 300, 350, 300)
        self.setWindowTitle('Vibration analysis')    
        self.show()
        # timer for callbacks, taken from:
        # http://ralsina.me/weblog/posts/BB974.html
        timer = QtCore.QTimer()
        timer.timeout.connect(self.handleNewData)
        
        # keep reference to timer        
        self.timer = timer
        
     
    def initData(self):
        mic = MicrophoneRecorder(rate=8000, chunksize=2048)

        # keeps reference to mic        
        self.mic = mic
        
        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize, 
                                         1./mic.rate)
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
        
        # Init spectrum processor
        self.sp = SpectrumProcessor()
        self.have_peak_plot = False
        
        # Init training frames dictionary
        self.trainingFrames = {}
        self.trainingLabelIndex = {}
        
        self.predict_frames = []
        

                
    def connectSlots(self):
        self.runButton.clicked.connect(self.runButtonClicked)
        self.resetButton.clicked.connect(self.resetButtonClicked)
        self.trainButton.clicked.connect(self.trainButtonClicked)

        
    def initMplWidget(self):
        """creates initial matplotlib plots in the main window and keeps 
        references for further use"""
        # top plot
        self.ax_top = self.main_figure.figure.add_subplot(311)
        self.ax_top.set_ylim(-32768, 32768)
        self.ax_top.set_xlim(0, self.time_vect.max())
        self.ax_top.set_xlabel(u'time (ms)', fontsize=6)

        # bottom plot
        self.ax_bottom = self.main_figure.figure.add_subplot(312)
        self.ax_bottom.set_ylim(0, 1)
        self.ax_bottom.set_xlim(0, self.freq_vect.max())
        self.ax_bottom.set_xlabel(u'frequency (Hz)', fontsize=6)
        
        # bottom plot
        self.ax_hist = self.main_figure.figure.add_subplot(313)        
        self.ax_hist.set_xlim(0,500)
        self.ax_hist.set_ylim(0,0.5)
        # line objects        
        self.line_top, = self.ax_top.plot(self.time_vect, 
                                         np.ones_like(self.time_vect))
        
        self.line_bottom, = self.ax_bottom.plot(self.freq_vect,
                                               np.ones_like(self.freq_vect))
                                               
    def runButtonClicked(self):
        if self.runButton.isChecked():
            self.mic.start() # Start microphone recorder
            self.timer.start(100)   # Start timer    

        else:
            self.timer.stop()  # Stop timer
            self.mic.stop()    # Stop microphone recorder  
              
    def resetButtonClicked(self):  
        self.trainingFrames = {}
        self.setTrainingFrameNumInTable(0)

    def trainButtonClicked(self):    
        
        try:
            # Init machine learning object
            mlModel = self.machineLearningModelCombo.currentText()
            print(mlModel)
            self.ml = MachineLearning(mlModel)
#            for k in self.trainingFrames.keys():
#                self.ml.AppendLabel(self.trainingLabelIndex[k])
#                self.ml.ExtractAndAppendFeatures(self.freq_vect,self.trainingFrames[k])
#            self.ml.Concatenate()   # Concatenate list of numpy arrays into one
            #print(self.ml.features.shape)
            #print(self.ml.labels.shape)
#            X = self.ml.features
#            y = self.ml.labels
            X = np.vstack(self.trainingFrames['Normal'])
            print('Training SVC anomaly model')
            #self.ml.clf.fit(X, y) # Training model
            self.ml.clf.fit(X)
            print('Done training')
            
        except Exception as e:
            print('error: {0}'.format(e))
            traceback.print_exc()
            raise
    
    def setTrainingFrameNumInTable(self,num,index,label):
        item = QtWidgets.QTableWidgetItem()
        item.setData(QtCore.Qt.DisplayRole,num)
        self.trainingTable.setItem(index, self.TRAINING_SIZE_COL, item)
        item2 = QtWidgets.QTableWidgetItem()
        item2.setText(label)
        self.trainingTable.setItem(index, self.LABEL_COL, item2)
        
                         
    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """ 
        # Check what mode we are in (radio button).
        # If in training mode:
        if self.trainRadioButton.isChecked():
            self.newDataTrainModeAnomaly()
        # If in prediction mode
        elif self.predictRadioButton.isChecked():
            self.newDataPredictMode()
            
    def newDataTrainModeAnomaly(self, avgwind_train = 5, num_features = 8):
        # 1) Get frames
        frames = self.mic.get_frames()
        sp = self.sp            
        
        try:
            if len(frames) > 0:
                # keeps only the last frame
                current_frame = frames[-1]
                # plots the time signal
                self.line_top.set_data(self.time_vect, current_frame)
                
                if self.recTrainButton.isChecked(): # Training button is on. Save all spectra for training purposes
                    # 2) Do frame fft. 
                    (freq_vect, fft_frames) = CalculateFramesFFT(frames, avgwind=avgwind_train)
                    # 3) Get features
                    X_norm = GetFeaturesFromFrames(freq_vect, fft_frames, num_features=num_features)
                    fft_frame = fft_frames[-1]
                    # Append them to feature array
                    #label = self.trainingLabelComboBox.currentText()
                    #index = self.trainingLabelComboBox.currentIndex()
                    label = 'Normal'
                    index = 0
                    if label in self.trainingFrames:
                        self.trainingFrames[label].extend(X_norm)
                        self.trainingLabelIndex[label].append(index)
                    else:
                        self.trainingFrames[label] = [X_norm]
                        self.trainingLabelIndex[label] = [index]
                    
                    self.setTrainingFrameNumInTable(len(self.trainingFrames[label]),index,label)
                else:
                    # computes and plots the fft signal            
                    fft_frame = np.fft.rfft(current_frame)
    
                if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                    fft_frame /= np.abs(fft_frame).max()
                else:
                    fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                    #print(np.abs(fft_frame).max())
                        
                self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame)) 
                
                min_freq = 90
                ok = self.freq_vect > min_freq
                amp = np.abs(fft_frame)
                
                sp.setData(self.freq_vect[ok],amp[ok])    
    
                #pi = sp.getPeaksFromSpectrogram(freq = self.freq_vect[ok].tolist(),amps = amp[ok].tolist(),peaksPerSpectrum = 10)
                pi = sp.findPeaksPU()
                sortedind = np.argsort(amp[pi])[::-1]
                
                peaksPerSpectrum = 10
                pks_freqs = sp.peak_freqs[sortedind[:peaksPerSpectrum]]
                pks_amps = sp.peak_values[sortedind[:peaksPerSpectrum]]
                #pks_freqs = sp.peak_freqs[pi]
                #pks_amps = sp.peak_values[pi]
                if not self.have_peak_plot:
                    self.have_peak_plot = True
                    self.ax_bottom.hold(True)
                    #self.peak_plot, = self.ax_bottom.plot(sp.peak_freqs[sortedind],sp.peak_values[sortedind],'xr')
                    self.peak_plot, = self.ax_bottom.plot(pks_freqs, pks_amps,'xr')
                    self.ax_bottom.hold(False)
                else:
                    #self.peak_plot.set_data(sp.peak_freqs[sortedind],sp.peak_values[sortedind])
                    self.peak_plot.set_data(pks_freqs, pks_amps)
                    
                
                
                
                self.main_figure.canvas.draw()
        except Exception as e:
            print('error: {0}'.format(e))
            traceback.print_exc()
            raise
            
    def newDataTrainMode(self):
        """ handle new data when in training mode"""
        # gets the latest frames        
        frames = self.mic.get_frames()
         
        try:
            if len(frames) > 0:
                # keeps only the last frame
                current_frame = frames[-1]
                # plots the time signal
                self.line_top.set_data(self.time_vect, current_frame)
                
                
                if self.recTrainButton.isChecked(): # Training button is on. Save all spectra for training purposes
                    fft_frames = np.array([np.fft.rfft(fr) for fr in frames])
                    if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                        fft_frames /= np.abs(fft_frames).max()
                    else:
                        fft_frames *= (1 + self.fixedGainSlider.value()) / 5000000.
                    fft_frame = fft_frames[-1]
                    
                    label = self.trainingLabelComboBox.currentText()
                    index = self.trainingLabelComboBox.currentIndex()
                    if label in self.trainingFrames:
                        self.trainingFrames[label].append(np.abs(fft_frames[0])) # Todo: avoid concatenation which makes code slow
                        self.trainingLabelIndex[label].append(index)
                    else:
                        self.trainingFrames[label] = [np.abs(fft_frames[0])]
                        self.trainingLabelIndex[label] = [index]
                        
                    self.setTrainingFrameNumInTable(len(self.trainingFrames[label]),index,label)
                    
                    
                else:
                    # computes and plots the fft signal            
                    fft_frame = np.fft.rfft(current_frame)
                    if self.autoGainCheckBox.checkState() == QtCore.Qt.Checked:
                        fft_frame /= np.abs(fft_frame).max()
                    else:
                        fft_frame *= (1 + self.fixedGainSlider.value()) / 5000000.
                        #print(np.abs(fft_frame).max())
                        
                self.line_bottom.set_data(self.freq_vect, np.abs(fft_frame))            
                
                # Process spectrum. Find fundamental frequency and show in text box
                sp = self.sp            
                sp.setData(self.freq_vect,np.abs(fft_frame))
    
                if self.getFundamentalCheckBox.checkState() == QtCore.Qt.Checked:
                    pitchDetectionAlgorithm = self.pitchDetectionAlgorithmCombo.currentText()
                    if pitchDetectionAlgorithm == 'Peak detection':
                        
                        fund, sortedhist, hist, edges = sp.getFundamentalByPeakDetect()
                        if hasattr(self, 'histplot'):
                            #self.histplot.set_data(0.5*(edges[:-1]+edges[1:]),hist)
                            for rect, h, e in zip(self.histplot, hist, edges):
                                rect.set_height(h)
                                rect.set_x(e)
                                rect.set_width(edges[1]-edges[0])
                                
                            xlim = self.ax_hist.get_xlim()
                            if min(edges) < min(xlim):
                                self.ax_hist.set_xlim(min(edges),xlim[1])
                            if max(edges) > max(xlim):
                                self.ax_hist.set_xlim(xlim[0],max(edges))
                        else:
                            
                            self.histplot = self.ax_hist.bar(0.5*(edges[:-1]+edges[1:]),hist)  
                            
                    elif pitchDetectionAlgorithm == 'Harmonic product spectrum':
                        
                        fund, aP, MI = sp.getFundamentalByHPS()
                        if not hasattr(self, 'HPSplot'):
                            
                            self.HPSplot, = self.ax_hist.plot(self.freq_vect[:len(aP)],aP)  
                            self.ax_hist.hold(True)
                            self.HPSpeakPlot, = self.ax_hist.plot(fund,aP[MI],'xr')
                            self.ax_hist.hold(False)
                        else:
                            self.HPSplot.set_data(self.freq_vect[:len(aP)],aP)
                            self.HPSpeakPlot.set_data(fund,aP[MI])
    
                    self.fundFreqText.setText('%-f' % (fund,))
                else:
                    sp.findPeaks()
                
                #print('{0} {1}'.format(sp.peak_freqs,sp.peak_values))
                
                
                
                if not self.have_peak_plot:
                    self.have_peak_plot = True
                    self.ax_bottom.hold(True)
                    self.peak_plot, = self.ax_bottom.plot(sp.peak_freqs,sp.peak_values,'xr')
                    self.ax_bottom.hold(False)
                else:
                    self.peak_plot.set_data(sp.peak_freqs,sp.peak_values)
                
                # refreshes the plots
                self.main_figure.canvas.draw()
                
        except Exception as e:
            print('error: {0}'.format(e))
            traceback.print_exc()
            raise
            
    def newDataPredictMode(self, avgwind_predict = 2, num_features = 8, num_frames_per_prediction = 20):
        frames = self.mic.get_frames()
        print(len(self.predict_frames))
        if len(self.predict_frames) == 0:
            self.predict_frames = frames
        else:
            self.predict_frames.extend(frames)
        if len(self.predict_frames) >= num_frames_per_prediction:
            frames = self.predict_frames
            #print(frames)
            #print(len(frames))
            self.predict_frames = []
            (freq_vect, fft_frames) = CalculateFramesFFT(frames, avgwind=avgwind_predict)
            Xtest = GetFeaturesFromFrames(freq_vect, fft_frames, num_features=num_features)
            prediction = self.ml.clf.predict(Xtest)
            #print(prediction)
            FailureProb = (sum(prediction==-1)/float(len(prediction)))
            output_text = 'Probability of failure: {0}'.format(FailureProb)
            self.predictionText.setText(output_text)
            #print(output_text)

        
import sys 

if __name__ == "__main__": 
    app = QtWidgets.QApplication(sys.argv) 
    window = VibrationAnalysisMainWindow() 
    sys.exit(app.exec_())