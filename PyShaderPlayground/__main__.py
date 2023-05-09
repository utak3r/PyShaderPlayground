__version__ = "0.1.0"

import sys
import os
import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QIcon
from PyShaderPlayground.main_window import ShaderPlayground

if __name__ == "__main__":

    dirname = os.path.dirname(PySide6.__file__)
    #dirname = 'g:\\Python\\Qt6\\Library'
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('app.ico'))

    mainWnd = ShaderPlayground()
    mainWnd.show()

    sys.exit(app.exec())
