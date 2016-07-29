import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtGui, QtCore, uic
from MplFigure import MplFigure
from MicrophoneRecorder import MicrophoneRecorder
from SpectrumProcessor import SpectrumProcessor, MachineLearning
#import seaborn
import traceback

Ui_MainWindow, QMainWindow = uic.loadUiType("VibrationGUI_V2.ui") 

class VibrationAnalysisMainWindow(QtGui.QMainWindow,Ui_MainWindow):
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
        mic = MicrophoneRecorder(rate=4000, chunksize=2048)

        # keeps reference to mic        
        self.mic = mic
        
        # computes the parameters that will be used during plotting
        self.freq_vect = np.fft.rfftfreq(mic.chunksize, 
                                         1./mic.rate)
        self.time_vect = np.arange(mic.chunksize, dtype=np.float32) / mic.rate * 1000
        
        # Init spectrum processor
        self.sp = SpectrumProcessor()
        self.have_peak_plot = False
        
        
        # Init training frame buffer
        self.trainingFrames = []
                
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
        self.trainingFrames = []
        self.setTrainingFrameNumInTable(0)

    def trainButtonClicked(self):      
        # Init machine learning object
        mlModel = self.machineLearningModelCombo.currentText()
        print(mlModel)
        self.ml = MachineLearning(mlModel)
        self.ml.ExtractFeatures(self.freq_vect,self.trainingFrames)
        
    def setTrainingFrameNumInTable(self,num):
        item = QtGui.QTableWidgetItem()
        item.setData(QtCore.Qt.DisplayRole,num)
        self.trainingTable.setItem(0, 0, item)
                                  
    def handleNewData(self):
        """ handles the asynchroneously collected sound chunks """        
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
                    if len(self.trainingFrames) > 0:
                        self.trainingFrames = np.concatenate(self.trainingFrames,np.abs(fft_frames),axis=0) # Todo: avoid concatenation which makes code slow
                    else:
                        self.trainingFrames = fft_frames
                        
                    self.setTrainingFrameNumInTable(len(self.trainingFrames))
                    
                    
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
                #print('sp\n')
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
            
import sys 

if __name__ == "__main__": 
    app = QtGui.QApplication(sys.argv) 
    window = VibrationAnalysisMainWindow() 
    sys.exit(app.exec_())