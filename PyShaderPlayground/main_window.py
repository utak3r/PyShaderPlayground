from PySide2.QtCore import QCoreApplication, Qt, Slot, QUrl, QFile, QIODevice, QFileInfo
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow, QSizePolicy
from PySide2.QtUiTools import QUiLoader
from PyShaderPlayground.opengl_widget import ShaderWidget
from PyShaderPlayground.text_tools import GLSLSyntaxHighlighter

class ShaderPlayground(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.init_ui("PyShaderPlayground/ShaderPlayground.ui")
        self.opengl = self.centralWidget().player

        self.syntax_highlighter = GLSLSyntaxHighlighter(self.centralWidget().txtShaderEditor.document())

        self.centralWidget().txtShaderEditor.setText(self.opengl.get_shader())
        self.centralWidget().btnCompile.clicked.connect(self.compile_shader)
        self.centralWidget().btnLoadFile.clicked.connect(self.open_shader_from_file)
        self.centralWidget().btnSaveFile.clicked.connect(self.save_shader_to_file)
        self.centralWidget().btnPlayPause.clicked.connect(self.play_pause_animation)
        self.centralWidget().btnRewind.clicked.connect(self.rewind_animation)

        self.current_filename = ""
        self.resize(1280, 720)

    def init_ui(self, filename):
        """ Read UI from file. """
        loader = U3UiLoader()
        loader.addPluginPath("PyShaderPlayground/")
        loader.registerCustomWidget(ShaderWidget)
        print(loader.availableWidgets())
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
        self.opengl.animation_play_pause()
    
    @Slot()
    def rewind_animation(self):
        self.opengl.animation_rewind()

class U3UiLoader(QUiLoader):
    """ Custom UiLoader, for loading up custom widgets. """
    def createWidget(self, className, parent=None, name=""):
        if className == "ShaderWidget":
            widget = ShaderWidget(parent)
        else:
            widget = super().createWidget(className, parent, name)
        return widget
