__version__ = "0.1.0"

import sys
from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QCoreApplication, Qt
from PySide2.QtGui import QIcon, QSurfaceFormat
from PyShaderPlayground.main_window import ShaderPlayground

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('app.ico'))

    # Setting up modern OpenGL format
    # OpenGL_format = QSurfaceFormat()
    # OpenGL_format.setDepthBufferSize(24)
    # OpenGL_format.setStencilBufferSize(8)
    # OpenGL_format.setVersion(1, 2)
    # OpenGL_format.setProfile(QSurfaceFormat.CoreProfile)
    # QSurfaceFormat.setDefaultFormat(OpenGL_format)

    mainWnd = ShaderPlayground()
    mainWnd.show()

    sys.exit(app.exec_())
