__version__ = "0.1.0"

import sys
import os
import PySide6
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication, Qt, QCommandLineOption, QCommandLineParser
from PySide6.QtGui import QIcon
from PyShaderPlayground.main_window import ShaderPlayground

def parse_command_line(app):
    parser = QCommandLineParser()
    parser.addHelpOption()
    parser.addVersionOption()
    parser.addOption(QCommandLineOption(['s', 'shader'], 'Load specified shader at startup.', 'filename', ''))
    parser.process(app)
    loadshader_filename = parser.value('shader')
    return (loadshader_filename)


if __name__ == "__main__":

    dirname = os.path.dirname(PySide6.__file__)
    #dirname = 'g:\\Python\\Qt6\\Library'
    plugin_path = os.path.join(dirname, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Shader Playground")
    app.setApplicationVersion("1.0")
    app.setWindowIcon(QIcon('app.ico'))

    loadshader_filename = parse_command_line(app)
    print(f'Requested shader for preloading: {loadshader_filename}')

    mainWnd = ShaderPlayground(loadshader_filename)
    mainWnd.show()

    sys.exit(app.exec())
