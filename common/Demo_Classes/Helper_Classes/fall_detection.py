from collections import deque
import copy

from PySide2 import QtCore, QtWidgets


# QSlider extended to make the mouse click event move the slider to the location of the mouse
class FallDetectionSliderClass(QtWidgets.QSlider):
    def mousePressEvent(self, event):
        super(FallDetectionSliderClass, self).mousePressEvent(event)
        if event.button() == QtCore.Qt.LeftButton:
            val = self.pixelPosToRangeValue(event.pos())
            self.setValue(val)

    def pixelPosToRangeValue(self, pos):
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        gr = self.style().subControlRect(QtWidgets.QStyle.CC_Slider,
                                         opt, QtWidgets.QStyle.SC_SliderGroove, self)
        sr = self.style().subControlRect(QtWidgets.QStyle.CC_Slider,
                                         opt, QtWidgets.QStyle.SC_SliderHandle, self)
        sliderLength = sr.width()
        sliderMin = gr.x()
        sliderMax = gr.right() - sliderLength + 1
        pr = pos - sr.center() + sr.topLeft()
        p = pr.x()
        return QtWidgets.QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), p - sliderMin, sliderMax - sliderMin, opt.upsideDown)

# Class meant to store the fall detection results of all the tracks


class FallDetection:

    # Initialize the class with the default parameters (tested empirically)
    def __init__(self, maxNumTracks=10, frameTime=55, fallingThresholdProportion=0.6, secondsInFallBuffer=2.5):
        self.fallingThresholdProportion = fallingThresholdProportion
        self.secondsInFallBuffer = secondsInFallBuffer
        self.heightHistoryLen = int(
            round(self.secondsInFallBuffer * frameTime))
        self.heightBuffer = [deque(
            [-5] * self.heightHistoryLen, maxlen=self.heightHistoryLen) for i in range(maxNumTracks)]
        self.tracksIDsInPreviousFrame = []
        # Fall results that will be displayed to screen
        self.fallBufferDisplay = [0 for i in range(maxNumTracks)]
        # How many frames do you want to display a fall on the screen for
        self.numFramesToDisplayFall = 100

    # Sensitivity as given by the FallDetectionSliderClass instance
    def setFallSensitivity(self, fallingThresholdProportion):
        self.fallingThresholdProportion = fallingThresholdProportion

    # Update the fall detection results for every track in the frame
    def step(self, heights, tracks):
        # Decrement results for fall detection display
        for idx, result in enumerate(self.fallBufferDisplay):
            self.fallBufferDisplay[idx] = max(
                self.fallBufferDisplay[idx] - 1, 0)

        trackIDsInCurrFrame = []
        # Populate heights for current tracks
        for height in heights:
            # Find track with correct TID
            for track in tracks:
                # Found correct track
                if (int(track[0]) == int(height[0])):
                    tid = int(height[0])
                    self.heightBuffer[tid].appendleft(height[1])
                    trackIDsInCurrFrame.append(tid)

                    # Check if fallen
                    if (self.heightBuffer[tid][0] < self.fallingThresholdProportion * self.heightBuffer[tid][-1]):
                        self.fallBufferDisplay[tid] = self.numFramesToDisplayFall

        # Reset the buffer for tracks that were detected in the previous frame but not the current frame
        tracksToReset = set(self.tracksIDsInPreviousFrame) - \
            set(trackIDsInCurrFrame)
        for track in tracksToReset:
            for frame in range(self.heightHistoryLen):
                # Fill the buffer with -5's to remove any history for the track
                self.heightBuffer[track].appendleft(-5)
        self.tracksIDsInPreviousFrame = copy.deepcopy(trackIDsInCurrFrame)

        return self.fallBufferDisplay


# TODO This stuff was never used in original implementation?
#     def resetFallText(self):
#         self.fallAlert.setText('Standing')
#         self.fallPic.setPixmap(self.standingPicture)
#         self.fallResetTimerOn = 0


#     def updateFallThresh(self):
#         try:
#             newThresh = float(self.fallThreshInput.text())
#             self.fallThresh = newThresh
#             self.fallThreshMarker.setPos(self.fallThresh)
#         except:
#             print('No numerical threshold')
