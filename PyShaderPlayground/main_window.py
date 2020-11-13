from PySide2.QtCore import QCoreApplication, Qt, Slot, QUrl, QFile, QIODevice, QFileInfo
from PySide2.QtWidgets import QApplication, QFileDialog, QMainWindow
from PySide2.QtUiTools import QUiLoader
from PyShaderPlayground.opengl_widget import ShaderWidget
from PyShaderPlayground.text_tools import GLSLSyntaxHighlighter

class ShaderPlayground(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.init_ui("PyShaderPlayground/ShaderPlayground.ui")

        self.opengl = ShaderWidget(800, 450, self.centralWidget().player)
        self.opengl.setParent(self.centralWidget().player)
        #self.centralWidget().player = ShaderWidget()
        #self.centralWidget().player.resize(800,450)
        #self.centralWidget().player.show()
        self.syntax_highlighter = GLSLSyntaxHighlighter(self.centralWidget().txtShaderEditor.document())
        self.opengl.show()

        self.centralWidget().txtShaderEditor.setText(self.opengl.get_shader())
        self.centralWidget().btnCompile.clicked.connect(self.compile_shader)
        self.centralWidget().btnLoadFile.clicked.connect(self.open_shader_from_file)
        self.centralWidget().btnSaveFile.clicked.connect(self.save_shader_to_file)

        self.current_filename = ""

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

    @Slot()
    def open_shader_from_file(self):
        filename = QFileDialog.getOpenFileName(self, "Open shader", ".", "Shader files (*.glsl)")
        if filename[0] != "":
            with open(filename[0]) as file:
                self.current_filename = filename[0]
                self.centralWidget().txtShaderEditor.setText(file.read())

    @Slot()
    def save_shader_to_file(self):
        if self.current_filename != "":
            filename = QFileDialog.getSaveFileName(self, "Save shader as...", self.current_filename, "Shader files (*.glsl)")
        else:
            filename = QFileDialog.getSaveFileName(self, "Save shader as...", "new_shader.glsl", "Shader files (*.glsl)")
        if filename[0] != "":
            self.current_filename = filename[0]
            with open(self.current_filename, 'w') as file:
                file.write(self.centralWidget().txtShaderEditor.toPlainText())
