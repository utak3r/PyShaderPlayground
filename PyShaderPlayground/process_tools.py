import asyncio
from PySide6.QtWidgets import QDialog
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QCoreApplication, Slot, QFile, QIODevice
from os import path

class ProcessRunner(QDialog):
    """ Running a shell command with stdout and stderr output in a text window. """
    def __init__(self):
        QDialog.__init__(self)
        self.init_ui(path.abspath(path.join(path.dirname(__file__), 'ProcessRunner.ui')))

        self.process = None
        self.Form.btnOk.clicked.connect(self.accept)
        self.Form.btnCancel.clicked.connect(self.terminate)

    def init_ui(self, filename):
        """ Init UI from a given ui file. """
        loader = QUiLoader()
        file = QFile(filename)
        file.open(QIODevice.ReadOnly)
        self.Form = loader.load(file, self)
        file.close()

    def run_command(self, command):
        """ Run a given command asynchronously. """
        self.show()
        self.raise_()
        self.activateWindow()
        self.add_log(command)
        QCoreApplication.processEvents()
        asyncio.run(self.run(command))

    async def read_stream(self, stream):
        """ Read line by line from a given stream. """
        while self.process.returncode is None:
            try:
                line = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue
            if line:
                line = line.decode("utf-8")[:-1]
                #print(line)
                self.add_log(line)
                QCoreApplication.processEvents()
            else:
                break

    async def run(self, command):
        """ Create a subprocess and let read_stream method to read its output. Change the dialog's buttons states. """
        self.Form.btnOk.setEnabled(False)
        try:
            self.process = await asyncio.create_subprocess_shell(command, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            self.Form.btnCancel.setEnabled(True)
            async with asyncio.TaskGroup() as tg:
                read_task = tg.create_task(self.read_stream(self.process.stdout))
        finally:
            self.Form.btnCancel.setEnabled(False)
            self.Form.btnOk.setEnabled(True)


    @Slot()
    def terminate(self):
        """ Terminate the subprocess. """
        if self.process is not None:
            self.process.terminate()
        self.reject()

    @Slot()
    def add_log(self, text):
        """ Add a line of text to an output window. """
        self.Form.logText.appendPlainText(text)
