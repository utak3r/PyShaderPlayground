from PySide2.QtGui import QOpenGLTexture, QImage, QPixmap
from PySide2.QtCore import Qt
from enum import Enum
from pathlib import Path

class TextureFilter(Enum):
    """ Enum for texture filtering. """
    FILTER_NEAREST = QOpenGLTexture.Nearest
    FILTER_LINEAR = QOpenGLTexture.Linear
    FILTER_NEAREST_MIPMAP_NEAREST = QOpenGLTexture.NearestMipMapNearest
    FILTER_NEAREST_MIPMAP_LINEAR = QOpenGLTexture.NearestMipMapLinear
    FILTER_LINEAR_MIPMAP_NEAREST = QOpenGLTexture.LinearMipMapNearest
    FILTER_LINEAR_MIPMAP_LINEAR = QOpenGLTexture.LinearMipMapLinear

class TextureWrapMode(Enum):
    """ Enum for texture wrapping. """
    WRAP_REPEAT = QOpenGLTexture.Repeat
    WRAP_MIRRORED_REPEAT = QOpenGLTexture.MirroredRepeat
    WRAP_CLAMP_TO_EDGE = QOpenGLTexture.ClampToEdge
    WRAP_CLAMP_TO_BORDER = QOpenGLTexture.ClampToBorder


class InputTexture():
    def __init__(self):
        self.texture_ = None
        self.filter_minification_ = TextureFilter.FILTER_LINEAR_MIPMAP_LINEAR
        self.filter_magnification_ = TextureFilter.FILTER_LINEAR
        self.wrapping_ = TextureWrapMode.WRAP_CLAMP_TO_EDGE
        self.filename_ = ""

    def get_texture(self) -> QOpenGLTexture:
        return self.texture_

    def create_texture(self):
        self.texture_ = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.texture_.setMinificationFilter(self.filter_minification_.value)
        self.texture_.setMagnificationFilter(self.filter_magnification_.value)
        self.texture_.setWrapMode(self.wrapping_.value)

    def get_thumbnail(self) -> QPixmap:
        pixmap = QPixmap(100, 100)
        pixmap.fill(Qt.black)
        return pixmap


#
# InputTexture2D
#
class InputTexture2D(InputTexture):
    def __init__(self, filename: str):
        super().__init__()
        self.wrapping_ = TextureWrapMode.WRAP_REPEAT
        self.create_texture(filename)

    def create_texture(self, filename: str):
        super().create_texture()
        self.filename_ = filename
        self.texture_.setData(QImage(filename).mirrored())

    def get_thumbnail(self) -> QPixmap:
        file = Path(self.filename_)
        if file.is_file():
            pixmap = QPixmap(QImage(self.filename_).scaled(100, 100, Qt.KeepAspectRatio))
        else:
            pixmap = QPixmap(100, 100)
            pixmap.fill(Qt.black)
        return pixmap


#
# InputTextureSound
#
class InputTextureSound(InputTexture):
    def __init__(self):
        super().__init__()

