from PySide6.QtWidgets import QDialog, QFileDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QCoreApplication, Slot, QFile, QIODevice, QSettings, QFileInfo
from enum import IntEnum
from os import path

class VideoEncodingParams(QDialog):

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.init_ui(path.abspath(path.join(path.dirname(__file__), 'VideoEncodingParams.ui')))

        self.settings = settings

        self.fill_resolutions_combobox()
        self.Form.cbxVideoResolutions.setEditable(False)
        self.Form.cbxVideoResolutions.currentIndexChanged.connect(self.select_resolution)
        self.Form.cbxVideoResolutions.setCurrentIndex(VideoResolutionStandard.RES_720P)
        self.setWindowTitle("Video encoding parameters")
        self.Form.buttonBox.accepted.connect(self.accept)
        self.Form.buttonBox.rejected.connect(self.reject)
        self.Form.layout().setContentsMargins(8, 8, 8, 8)
        self.Form.edWidth.valueChanged.connect(lambda val: self.res_values_changed(True, False, False))
        self.Form.edHeight.valueChanged.connect(lambda val: self.res_values_changed(False, True, False))
        self.Form.cbxKeepAspectRatio.stateChanged.connect(lambda val: self.res_values_changed(False, False, True))

        self.settings.beginGroup("VideoEncoding")
        self.Form.ffmpeg.setText(settings.value("FFmpegPath", "ffmpeg.exe"))
        self.Form.Duration.setValue(int(settings.value("Duration", 5)))
        self.Form.Framerate.setValue(int(settings.value("Framerate", 30)))
        self.fill_codecs()
        self.Form.Codec.setCurrentText(self.settings.value("LastCodec", "-c:v libx264 -preset medium -tune animation"))
        self.settings.endGroup()
        self.Form.btnFindFFmpeg.clicked.connect(self.browse_for_ffmpeg_exec)



    def init_ui(self, filename):
        """ Init UI from a given ui file. """
        loader = QUiLoader()
        file = QFile(filename)
        file.open(QIODevice.ReadOnly)
        self.Form = loader.load(file, self)
        file.close()


    @Slot()
    def accept(self):
        """ Save settings and close the dialog with 'accepted' signal. """
        self.settings.beginGroup("VideoEncoding")
        self.settings.setValue("FFmpegPath", self.get_ffmpeg())
        self.settings.setValue("Duration", self.get_duration())
        self.settings.setValue("Framerate", self.get_framerate())
        self.settings.setValue("LastCodec", self.get_codec())
        self.settings.endGroup()
        self.settings.sync()
        super().accept()


    def fill_resolutions_combobox(self):
        """ Fill few default video resolutions. """
        self.Form.cbxVideoResolutions.clear()
        res = VideoResolution("240p", 352, 240)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("360p", 480, 360)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("480p", 858, 480)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("720p", 1280, 720)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("FullHD 1080p", 1920, 1080)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("2K 1152p", 2048, 1152)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("2K DCI Full", 2048, 1080)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("UltraHD 2160p", 3860, 2160)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("4K DCI Full", 4096, 2160)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())
        res = VideoResolution("8K UHD", 7680, 4320)
        self.Form.cbxVideoResolutions.addItem(res.name, res.get_user_data())


    def fill_codecs(self):
        """ Fill few default codec's parameters. """
        self.Form.Codec.clear()
        self.Form.Codec.addItem("-c:v libx264 -preset medium -tune animation")
        self.Form.Codec.addItem("-c:v dnxhd -b:v 185M")
        self.Form.Codec.addItem("-c:v prores_ks -profile:v 3 -vendor ap10 -pix_fmt yuv422p10le")
        self.Form.Codec.setEditable(True)
        self.Form.Codec.setCurrentIndex(0)


    @Slot()
    def select_resolution(self, index: int):
        """ Translates current resolution from a combobox into spinboxes. """
        res = self.get_current_resolution()
        self.Form.edWidth.setValue(res[1])
        self.Form.edHeight.setValue(res[2])

    def get_current_resolution(self):
        """ Returns current video resolution as data. """
        res = self.Form.cbxVideoResolutions.currentData()
        #print(res)
        return (res)

    def get_width(self):
        """ Returns current video width. """
        return self.Form.edWidth.value()

    def get_height(self):
        """ Returns current video height. """
        return self.Form.edHeight.value()
    
    def get_ffmpeg(self):
        """ Returns current FFmpeg's path. """
        return self.Form.ffmpeg.text()
    
    def get_duration(self):
        """ Returns current duration. """
        return self.Form.Duration.value()
    
    def set_duration(self, value):
        """ Sets new duration. """
        self.Form.Duration.setValue(int(value))
    
    def get_framerate(self):
        """ Returns current framerate. """
        return self.Form.Framerate.value()
    
    def get_codec(self):
        """ Returns current codec params. """
        return self.Form.Codec.currentText()

    @Slot()
    def res_values_changed(self, is_width_changed: bool, is_height_changed: bool, is_aspect_checked: bool):
        """ Maintaining aspect ratio for resolution. """
        is_aspect = self.Form.cbxKeepAspectRatio.isChecked()
        width = self.Form.edWidth.value()
        height = self.Form.edHeight.value()
        self.Form.edWidth.blockSignals(True)
        self.Form.edHeight.blockSignals(True)
        if is_aspect_checked and is_aspect:
            # user just checked "Keep aspect", so save current aspect ratio
            self.render_aspect_ratio = width / height
        if is_width_changed and is_aspect:
            # user changed width, and wants to keep aspect ratio, so change the height
            self.Form.edHeight.setValue(width / self.render_aspect_ratio)
        if is_height_changed and is_aspect:
            # user changed height, and wants to keep aspect ratio, so change the width
            self.Form.edWidth.setValue(height * self.render_aspect_ratio)
        self.Form.edWidth.blockSignals(False)
        self.Form.edHeight.blockSignals(False)

    @Slot()
    def browse_for_ffmpeg_exec(self):
        """Dialog for pointing to the ffmpeg executable"""
        filename = QFileDialog.getOpenFileName(self, "Find FFmpeg executable", QFileInfo(self.Form.ffmpeg.text()).absolutePath(), "Executables (*.exe)")
        if filename[0] != "":
            self.Form.ffmpeg.setText(filename[0])



class VideoResolutionStandard(IntEnum):
    """ Enum for video resolutions. """
    RES_240P = 0
    RES_360P = 1
    RES_480P = 2
    RES_720P = 3
    RES_FULLHD = 4
    RES_2K = 5
    RES_2K_DCI = 6
    RES_ULTRAHD = 7
    RES_4K_DCI = 8
    RES_8KUHD = 9


class VideoResolution():
    """ Class for defining and storing video resolution. """
    def __init__(self, name: str, width: int, height: int):
        self.define_resolution(name, width, height)

    def get_size(self):
        """ Returns resolution as a tuplet. """
        return (self.width, self.height)

    def define_resolution(self, name: str, width: int, height: int):
        """ Set given params. """
        self.name = name
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return self.name + ": " + str(self.width) + "x" + str(self.height)

    def get_user_data(self):
        """ Returns this reset as a UserData element, ready for inserting into a QComboBox list. """
        return (self.name, self.width, self.height)
