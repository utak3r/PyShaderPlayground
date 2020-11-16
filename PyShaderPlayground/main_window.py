from PySide2.QtCore import QCoreApplication, Qt, Slot, Signal, QUrl, QFile, QIODevice, QFileInfo
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow, QSizePolicy, QDialog, QSlider
from PySide2.QtUiTools import QUiLoader
from PyShaderPlayground.opengl_widget import ShaderWidget
from PyShaderPlayground.text_tools import GLSLSyntaxHighlighter

class ShaderPlayground(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.init_ui("PyShaderPlayground/ShaderPlayground.ui")
        self.opengl = self.centralWidget().player
        self.setWindowTitle("Shader Playground")

        self.syntax_highlighter = GLSLSyntaxHighlighter(self.centralWidget().txtShaderEditor.document())

        self.centralWidget().txtShaderEditor.setText(self.opengl.get_shader())
        self.centralWidget().btnCompile.clicked.connect(self.compile_shader)
        self.centralWidget().btnLoadFile.clicked.connect(self.open_shader_from_file)
        self.centralWidget().btnSaveFile.clicked.connect(self.save_shader_to_file)
        self.centralWidget().btnPlayPause.clicked.connect(self.play_pause_animation)
        self.centralWidget().btnRewind.clicked.connect(self.rewind_animation)
        self.centralWidget().btnSaveImage.clicked.connect(self.save_image)
        self.centralWidget().btnRecordAnimation.setEnabled(False)
        self.centralWidget().AnimationSlider.valueUpdated.connect(self.change_animation)

        self.current_filename = ""
        self.resize(1280, 540)
        if self.opengl.is_playing():
            self.centralWidget().btnPlayPause.setText("Pause")
        else:
            self.centralWidget().btnPlayPause.setText("Play")
        self.last_render_size = [1920, 1080]
        self.render_aspect_ratio = self.last_render_size[0] / self.last_render_size[1]


    def init_ui(self, filename):
        """ Read UI from file. """
        loader = U3UiLoader()
        loader.addPluginPath("PyShaderPlayground/")
        loader.registerCustomWidget(ShaderWidget)
        #print(loader.availableWidgets())
        file = QFile(filename)
        file.open(QIODevice.ReadOnly)
        self.setCentralWidget(loader.load(file, self))
        file.close()

    @Slot()
    def compile_shader(self):
        """ Compile and link a new shader. """
        shader = self.centralWidget().txtShaderEditor.toPlainText()
        self.opengl.set_shader(shader)

    @Slot()
    def open_shader_from_file(self):
        """ Load a shader from a file into editor. """
        filename = QFileDialog.getOpenFileName(self, "Open shader", ".", "Shader files (*.glsl)")
        if filename[0] != "":
            with open(filename[0]) as file:
                self.current_filename = filename[0]
                self.centralWidget().txtShaderEditor.setText(file.read())

    @Slot()
    def save_shader_to_file(self):
        """ Save from editor to a file. """
        if self.current_filename != "":
            filename = QFileDialog.getSaveFileName(self, "Save shader as...", 
                self.current_filename, "Shader files (*.glsl)")
        else:
            filename = QFileDialog.getSaveFileName(self, "Save shader as...", 
                "new_shader.glsl", "Shader files (*.glsl)")
        if filename[0] != "":
            self.current_filename = filename[0]
            with open(self.current_filename, 'w') as file:
                file.write(self.centralWidget().txtShaderEditor.toPlainText())

    @Slot()
    def play_pause_animation(self):
        """ Play/Pause animation. """
        self.opengl.animation_play_pause()
        if self.opengl.is_playing():
            self.centralWidget().btnPlayPause.setText("Pause")
        else:
            self.centralWidget().btnPlayPause.setText("Play")
    
    @Slot()
    def rewind_animation(self):
        """ Rewind an animation. Doesn't change the playing state. """
        self.opengl.animation_rewind()

    @Slot()
    def change_animation(self, value):
        """ User rubs the animation slider back or forth. """
        modifier = 1.0
        if value != 0:
            modifier = float(value) / 10.0
        self.opengl.set_animation_speed_modifier(modifier)
    
    @Slot()
    def save_image(self):
        """ Save current state as an image. """
        filename = QFileDialog.getSaveFileName(self, "Save image as...", 
            "render_image", "Image Files (*.png *.jpg)")
        if filename[0] != "":
            resolution_dialog = QDialog(self)
            ui_loader = QUiLoader()
            ui_file = QFile("PyShaderPlayground/ResolutionDialog.ui")
            ui_file.open(QIODevice.ReadOnly)
            resolution_dialog.Form = ui_loader.load(ui_file, resolution_dialog)
            ui_file.close()
            resolution_dialog.setWindowTitle("Set image resolution:")
            resolution_dialog.Form.buttonBox.accepted.connect(resolution_dialog.accept)
            resolution_dialog.Form.buttonBox.rejected.connect(resolution_dialog.reject)
            resolution_dialog.Form.layout().setContentsMargins(8, 8, 8, 8)
            resolution_dialog.Form.edWidth.setValue(self.last_render_size[0])
            resolution_dialog.Form.edHeight.setValue(self.last_render_size[1])
            resolution_dialog.Form.edWidth.valueChanged.connect(lambda val: self.resolution_dlg_value_changed(resolution_dialog, True, False, False))
            resolution_dialog.Form.edHeight.valueChanged.connect(lambda val: self.resolution_dlg_value_changed(resolution_dialog, False, True, False))
            resolution_dialog.Form.cbxKeepAspectRatio.stateChanged.connect(lambda val: self.resolution_dlg_value_changed(resolution_dialog, False, False, True))
            
            if QDialog.Accepted == resolution_dialog.exec():
                width = resolution_dialog.Form.edWidth.value()
                height = resolution_dialog.Form.edHeight.value()
                self.last_render_size = [width, height]
                self.opengl.render_image(filename[0], width, height)

    @Slot()
    def resolution_dlg_value_changed(self, dlg, is_width_changed: bool, is_height_changed: bool, is_aspect_checked: bool):
        """ Maintaining aspect ratio for resolution dialog. """
        is_aspect = dlg.Form.cbxKeepAspectRatio.isChecked()
        width = dlg.Form.edWidth.value()
        height = dlg.Form.edHeight.value()
        dlg.Form.edWidth.blockSignals(True)
        dlg.Form.edHeight.blockSignals(True)
        if is_aspect_checked and is_aspect:
            # user just checked "Keep aspect", so save current aspect ratio
            self.render_aspect_ratio = width / height
        if is_width_changed and is_aspect:
            # user changed width, and wants to keep aspect ratio, so change the height
            dlg.Form.edHeight.setValue(width / self.render_aspect_ratio)
        if is_height_changed and is_aspect:
            # user changed height, and wants to keep aspect ratio, so change the width
            dlg.Form.edWidth.setValue(height * self.render_aspect_ratio)
        dlg.Form.edWidth.blockSignals(False)
        dlg.Form.edHeight.blockSignals(False)


class U3UiLoader(QUiLoader):
    """ Custom UiLoader, for loading up custom widgets. """
    def createWidget(self, className, parent=None, name=""):
        if className == "ShaderWidget":
            widget = ShaderWidget(parent)
        elif className == "SpringSlider":
            widget = SpringSlider(parent)
        else:
            widget = super().createWidget(className, parent, name)
        return widget

class SpringSlider(QSlider):
    """ Spring recoil slider. """

    valueUpdated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(-100, 100)
        self.setSliderPosition(0)
        self.setTracking(True)
        self.sliderReleased.connect(self.sliderRelease)
        self.valueChanged.connect(self.valueUpdate)

    def sliderRelease(self):
        """ If user released the slider, bring it back to 0. """
        self.setValue(0)

    def valueUpdate(self, value):
        """ If slider is grabbed, just pass the value. If not (it was clicked), set to 0. """
        if not self.isSliderDown():
            self.setValue(0)
        self.valueUpdated.emit(value)
