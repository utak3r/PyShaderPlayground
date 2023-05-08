__version__ = "0.1.0"

import sys
import os
import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QIcon
from PyShaderPlayground.main_window import ShaderPlayground

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('app.ico'))

    mainWnd = ShaderPlayground()
    mainWnd.show()

    sys.exit(app.exec_())
