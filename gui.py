# test.py

"""Test GUI for Pollination Detector using PyQt6."""
import sys
# import os
import cv2
import csv
import numpy as np
import json
from timeit import default_timer
from invert_detection import (
    InvertDetector, rois_to_snippets, roi_to_snippets, closest_roi_point_is_within
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QStyle,
    QPushButton, QSlider,  QProgressBar, QSpinBox,
    QMainWindow, QStatusBar, QToolBar,
    QFileDialog,
    QVBoxLayout, QHBoxLayout, QTableView, 
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QDir, QUrl, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem

SP_MediaPlay = QStyle.SP_MediaPlay
SP_MediaPause = QStyle.SP_MediaPause
RIGHT = Qt.MouseButton.RightButton
LEFT = Qt.MouseButton.LeftButton


VIDEOS = ['raw.mp4', 'norm.mp4', 'mask.mp4']


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pollination Event Tagger")
        self._createMainScreen()
        self._createMenu()
        self._createToolBar()
        self._createStatusBar()
        self.blur = 5
        self.open = 5

    def _createMainScreen(self):
        self.player = VideoPlayer()
        self.table = QTableView()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectRows)

        layout = QHBoxLayout()
        layout.addWidget(self.player)
        layout.addWidget(self.table)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def _createMenu(self):
        menu = self.menuBar().addMenu("&File")
        menu.addAction("&Open Video", self.openVideo)
        menu.addAction("&Save", self.saveState)
        menu.addAction("&Load", self.loadState)
        menu.addAction("&Export", self.exportToCSV)

    def _createToolBar(self):
        tools = QToolBar()
        tools.addAction("Set ROIs", self.setROI)
        self.radiusInput = QSpinBox()
        self.radiusInput.setPrefix('Next ROI Radius: ')
        self.radiusInput.setSuffix(' px')
        self.radiusInput.setValue(10)
        self.radiusInput.setMinimum(1)
        self.radiusInput.valueChanged.connect(self._radiusChanged)
        tools.addWidget(self.radiusInput)
        self.blurInput = QSpinBox()
        self.blurInput.setPrefix('Blur Size: ')
        self.blurInput.setSuffix(' px')
        self.blurInput.setValue(5)
        self.blurInput.setMinimum(2)
        self.blurInput.valueChanged.connect(self._blurChanged)
        self.openInput = QSpinBox()
        self.openInput.setPrefix('Open Threshold: ')
        self.openInput.setSuffix(' px')
        self.openInput.setValue(5)
        self.openInput.setMinimum(1)
        self.openInput.valueChanged.connect(self._openChanged)
        tools.addWidget(self.blurInput)
        tools.addWidget(self.openInput)
        tools.addAction("AutoTag", self.autoTag)
        self.addToolBar(tools)

    def _radiusChanged(self, value):
        self.player.maskWidget.roi_radius = value

    def _blurChanged(self, value):
        self.blur = value

    def _openChanged(self, value):
        self.open = value

    def _createStatusBar(self):
        self.status = QStatusBar()
        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.status.addPermanentWidget(self.progressBar)
        self.setStatusBar(self.status)
        self.player.mediaPlayer.errorChanged.connect(self.handleError)
        self.status.showMessage("Open a video!")

    def openVideo(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open Video", QDir.homePath())
        if self.player.openVideo(file):
            self.video = file

    def saveState(self):
        self._save(QFileDialog.getSaveFileName(self, "Save Settings", QDir.homePath())[0])

    def _save(self, file=None):
        if file is None:
            file = self.file
        if file != '':
            self.file = file
            data = {'video_file': self.video, 'rois': self.player.maskWidget.rois,
                    'video_contours': {} if not hasattr(self.player.maskWidget, 'video_contours') else self.player.maskWidget.video_contours,
                    'collision_snippets': {} if not hasattr(self, 'model') else {ROI_to_str(k): v for k, v in self.model._data.items()},
                    'blur': self.blur, 'open': self.open}
            with open(file, 'w') as fp:
                json.dump(data, fp, cls=NumpyEncoder)
            print('done saving')

    def loadState(self):
        self._load(QFileDialog.getOpenFileName(self, "Open Settings", QDir.homePath())[0])

    def _load(self, file=None):
        if file is None:
            file = self.file
        if file != '':
            self.file = file
            with open(file, 'r') as fp:
                data = json.load(fp)
            if self.player.openVideo(data['video_file']):
                self.video = data['video_file']
            self.player.maskWidget.rois = data['rois']
            self.player.maskWidget.video_contours = {int(k): list_to_contours(v) for k, v in data['video_contours'].items()}
            if len(self.player.maskWidget.video_contours.items()) > 0:
                self.player.maskWidget.tagged = True
            self.setTable({str_to_ROI(k): v for k, v in data['collision_snippets'].items()})
            self.blurInput.setValue(data['blur'])
            self.openInput.setValue(data['open'])
            return True
        return False

    def exportToCSV(self, output):
        if self.player.maskWidget.tagged:
            if len(self.player.maskWidget.rois) > 0:
                self._events_to_CSV(self.model._data, output)

    def _events_to_CSV(self, events, output):
        fields = ['ROI', 'Event Frames']
        rows = []
        for roi, snippets in events.items():
            if snippets is not None:
                rows += [[roi, snippet] for snippet in snippets]
            with open(output, 'w') as f:
                write = csv.writer(f)
                write.writerow(fields)
                write.writerows(rows)

    def setROI(self):
        self.player.maskWidget.drawing = not self.player.maskWidget.drawing
        self.player.maskWidget.drawOverlay(self.player.maskWidget.position)

    def autoTag(self):
        self.status.showMessage(f"Autotagging {self.video}")
        self.detector = InvertDetector(self.video)
        self.detector.set_params(self.blur, self.open)      
        self.progressBar.setRange(0, self.detector.TOTAL-1)
        start = default_timer()
        for i, _raw, _norm, _mask in self.detector.process():
            self.progressBar.setValue(i)
        duration = default_timer() - start
        self.player.maskWidget.video_contours = self.detector.data
        # print(self.player.maskWidget.data)
        self.player.maskWidget.tagged = True
        self.setTable(rois_to_snippets(self.player.maskWidget.rois, self.player.maskWidget.video_contours))
        self.status.showMessage(f"Autotagging completed in {duration}s at {self.detector.TOTAL/duration}frames/s")

    def setTable(self, data):
        # print(data)
        self.model = InvertModel(data)
        self.table.setModel(self.model)
        self.table.clicked.connect(self.skip_to_event)

    def skip_to_event(self, item):
        _frame = int(item.siblingAtColumn(0).data())
        # _center = item.siblingAtColumn(1).data()
        # _center = (int(_center.split(', ')[0].split('(')[1]), int(_center.split(', ')[1].split(')')[0]))
        self.player.setPosition(int(round(66.67*_frame)))
        self.status.showMessage(f"{item.row()}, {_frame}")
        # self.player.draw_circle(_center[0], _center[1], 5, line_color=Qt.blue)

    def exit(self):
        sys.exit(app.exec())
 
    def handleError(self):
        self.player.playButton.setEnabled(False)
        self.status.showMessage("ERROR: " + self.player.mediaPlayer.errorString())

    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        self.player.resizeVideo()


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Orientation.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        self.scene = QGraphicsScene()
        self.videoWidget = QGraphicsVideoItem()
        self.maskWidget = MaskItem()
        self.scene.addItem(self.videoWidget)
        self.scene.addItem(self.maskWidget)
        self.graphicsView = QGraphicsView()
        self.graphicsView.setScene(self.scene)

        layout = QVBoxLayout()
        layout.addWidget(self.graphicsView)
        layout.addLayout(controlLayout)
        self.setLayout(layout)

        self.mediaPlayer = QMediaPlayer(None)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.playingChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.videoWidget.nativeSizeChanged.connect(self.resizeVideo)

    def openVideo(self, video: str) -> bool:
        if video != '':
            self.mediaPlayer.setSource(QUrl.fromLocalFile(video))
            self.playButton.setEnabled(True)
            self.mediaPlayer.pause()
            return True
        else:
            return False

    def play(self):
        if self.mediaPlayer.isPlaying():
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.isPlaying():
            self.playButton.setIcon(self.style().standardIcon(SP_MediaPause))
        else:
            self.playButton.setIcon(self.style().standardIcon(SP_MediaPlay))
 
    def positionChanged(self, position):
        self.positionSlider.setValue(position)
        self.maskWidget.drawOverlay(int(round(position/66.667)))
 
    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)
    
    def resizeVideo(self):
        self.videoWidget.setSize(self.videoWidget.nativeSize())
        self.maskWidget.height = int(self.videoWidget.nativeSize().height())
        self.maskWidget.width = int(self.videoWidget.nativeSize().width())
        if self.maskWidget.height > 0 and self.maskWidget.width > 0:
            image = np.zeros((self.maskWidget.height, self.maskWidget.width, 4),
                             dtype=np.uint8)
            if self.maskWidget.drawing:
                image[:] = (0, 255, 0, 128)
            self.maskWidget.updateImage(image)
            self.maskWidget.drawOverlay(self.maskWidget.position)
        self.graphicsView.fitInView(self.videoWidget.boundingRect(), Qt.KeepAspectRatio)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)


class MaskItem(QGraphicsPixmapItem):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.height = 0
        self.width = 0
        self.img = None
        self.drawing = False
        self.tagged = False
        self.rois = []
        self.position = 0
        self.roi_radius = 10

    def updateImage(self, cvImg):
        self.img = cvImg
        qtImg = self.cv2qt(cvImg)
        self.setPixmap(qtImg)

    def cv2qt(self, cvImg):
        h, w, ch = cvImg.shape
        bytesPerLine = ch*w
        p = QImage(cvImg.data, w, h, bytesPerLine, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(p)

    def drawROIs(self):
        for roi in self.rois:
            cv2.circle(self.img, roi[0], roi[1], (0, 0, 255, 255), 2)

    def drawROILabels(self):
        for i, roi in enumerate(self.rois):
            cv2.putText(self.img, f'{i+1}: {roi[0]}', roi[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255, 255), 2, cv2.LINE_AA)

    def drawOverlay(self, position):
        self.position = position
        self.img = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        if self.drawing:
            self.img[:] = (0, 255, 0, 128)
        if len(self.rois) > 0:
            self.drawROIs()
        if self.tagged:
            cv2.drawContours(self.img, self.video_contours[position], -1, (255, 0, 0, 255), 2)
        if len(self.rois) > 0:
            self.drawROILabels()
        qtImg = self.cv2qt(self.img)
        self.setPixmap(qtImg)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.drawing:
            if event.button() is LEFT:
                self.rois += [((int(event.pos().x()), int(event.pos().y())), self.roi_radius)]
            if event.button() is RIGHT:
                o = closest_roi_point_is_within((int(event.pos().x()), int(event.pos().y())), self.rois)
                if o >= 0:
                    del self.rois[o]
            # print(self.rois[-1])
            self.drawOverlay(self.position)


class InvertModel(QAbstractTableModel):
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data
        # print(self._data)
        self.unspooled = []
        for roi, snippets in self._data.items():
            if snippets is not None:
                self.unspooled += [[roi, snippet] for snippet in snippets]

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self.unspooled)
        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return 2
        return 0

    def addROI(self, roi, contours_in_video):
        self._data[roi] = roi_to_snippets(roi, contours_in_video)
        self.unspooled += [[roi, snippet] for snippet in self._data[roi]]

    def removeROI(self, roi):
        self._data.pop(roi, None)
        self.unspooled = [item for item in self.unspooled if item[0] == roi]

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return str(self.unspooled[index.row()][index.column()])
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section == 0:
                    return 'ROI'
                if section == 1:
                    return 'Collision Frames'
                return '???'
            if orientation == Qt.Vertical:
                return str(section)
        return None


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def ROI_to_str(roi):
    return f"{roi[0][0]},{roi[0][1]},{roi[1]}"


def str_to_ROI(s):
    temp = s.split(',')
    return ((int(temp[0]), int(temp[1])), int(temp[2]))


def list_to_contours(contours):
    return [np.array(contour, dtype=np.int32) for contour in contours]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.showMaximized()
    sys.exit(app.exec())
