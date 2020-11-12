from PySide2.QtCore import QCoreApplication, Qt, Slot, QUrl, QFile, QIODevice, QFileInfo
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow
from PySide2.QtUiTools import QUiLoader
from PyShaderPlayground.opengl_widget import ShaderWidget

class ShaderPlayground(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.init_ui("PyShaderPlayground/ShaderPlayground.ui")

        self.opengl = ShaderWidget(800, 450, self.centralWidget().player)
        self.opengl.setParent(self.centralWidget().player)
        #self.centralWidget().player = ShaderWidget()
        #self.centralWidget().player.resize(800,450)
        #self.centralWidget().player.show()
        self.opengl.show()
        self.centralWidget().btnCompile.clicked.connect(self.compile_shader)

    def init_ui(self, filename):
        loader = QUiLoader()
        file = QFile(filename)
        file.open(QIODevice.ReadOnly)
        self.setCentralWidget(loader.load(file, self))
        file.close()

    @Slot()
    def compile_shader(self):
        shader = self.centralWidget().txtShaderEditor.toPlainText()
        self.opengl.set_shader(shader)
