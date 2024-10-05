from PySide6.QtGui import QImage, QPixmap
from PySide6.QtOpenGL import QOpenGLTexture
from PySide6.QtCore import Qt
from enum import Enum
from pathlib import Path
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors as clrs
from skimage.transform import resize
from skimage import exposure


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

    def can_be_binded(self):
        return False

    def set_position(self, position: float):
        return True

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

    def can_be_binded(self):
        return True


#
# InputTextureSound
#
class InputTextureSound(InputTexture):
    def __init__(self, filename: str):
        super().__init__()
        self.audio_ = None
        self.sample_rate_ = 1
        self.framerate_ = 1
        self.length_ = 0
        self.max_sample_value_ = 0
        self.current_frame_ = 0
        self.thumbnail_ = None
        self.colormap_ = InputTextureSound.create_color_map()
        self.create_texture(filename)

    @staticmethod
    def create_color_map():
        reds = plt.get_cmap('Reds', 256)
        black_reds = reds(np.linspace(0, 1, 256))
        for i in range(256):
            black_reds[i:i+1, :] = np.array([i/256, 0, 0, 1])
        map_black_reds = clrs.ListedColormap(black_reds)
        return map_black_reds

    def create_texture(self, filename: str):
        super().create_texture()
        self.filename_ = filename
        self.sample_rate_, self.audio_ = wav.read(self.filename_)
        if self.audio_.ndim > 1:
            self.audio_ = np.mean(self.audio_, axis=1) # we want mono!
        num_samples = self.audio_.shape[0]
        self.length_ = num_samples / self.sample_rate_
        self.max_sample_value_ = np.max(self.audio_)

        fig=plt.figure(figsize=(1.0, 1.0), dpi=100)
        canvas = FigureCanvas(fig)
        ax = plt.axes()
        ax.set_axis_off()
        ax.margins(0)
        ax.plot(np.arange(num_samples) / self.sample_rate_, self.audio_)
        fig.tight_layout()
        thumb_file = f"{self.filename_}_thumbnail_temp.png"

        canvas.draw()
        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(100, 100, 3)
        img2 = QImage(img.data, 100, 100, QImage.Format_Indexed8)

        #plt.savefig(thumb_file, dpi=fig.dpi)
        plt.close(fig)
        #self.thumbnail_ = QPixmap(QImage(thumb_file))
        self.thumbnail_ = QPixmap(img2)
        #Path(thumb_file).unlink()
        self.texture_.setData(self.prepare_texture(0.0))

    def prepare_texture(self, position: float):
        current_frame = int(position*self.framerate_)
        sample_start = int(current_frame / self.framerate_ * self.sample_rate_)
        sample_end = int((current_frame + 1) / self.framerate_ * self.sample_rate_)
        N = sample_end - sample_start
        T = 1.0 / self.sample_rate_
        audio_part = self.audio_[sample_start:sample_end]

        fig=plt.figure(figsize=(5.12, 0.02), dpi=100)
        fig.subplots_adjust(hspace=0)
        fig.tight_layout()
        axes=[]
        axes.append(fig.add_subplot(2, 1, 1))
        axes[0].set_axis_off()
        axes[0].margins(0)
        axes.append(fig.add_subplot(2, 1, 2))
        axes[1].set_axis_off()
        axes[1].margins(0)

        Pxx, freqs, bins, im0 = axes[0].specgram(audio_part, Fs=self.framerate_, NFFT=1024, cmap=self.colormap_)
        image0 = im0.make_image(plt.gcf().canvas.get_renderer())
        image0 = np.array(image0, dtype=np.uint8)
        h, w = image0.shape
        texture = QImage(image0.data, h, w, 3*h, QImage.Format_RGB888)
        texture.save("texture_test.jpg")

        # spectrum = fft(audio_part, axis=0)
        # spectrum = np.abs(spectrum[:N//2])

        # audio_part_img = exposure.rescale_intensity(audio_part, out_range=(-1.0, 1.0))
        # audio_part_img = np.expand_dims(audio_part_img, axis=0)
        # audio_part_img = resize(audio_part_img, (1, 512), anti_aliasing=True)

        # spectrum_img = exposure.rescale_intensity(spectrum, out_range=(0, 100))
        # spectrum_img = np.expand_dims(spectrum_img, axis=0)
        # spectrum_img = resize(spectrum_img, (1, 512), anti_aliasing=True)

        # fig=plt.figure(figsize=(5.12, 0.02), dpi=100)
        # fig.subplots_adjust(hspace=0)
        # fig.tight_layout()
        # axes=[]
        # axes.append(fig.add_subplot(2, 1, 1))
        # axes[0].set_axis_off()
        # axes[0].margins(0)
        # axes.append(fig.add_subplot(2, 1, 2))
        # axes[1].set_axis_off()
        # axes[1].margins(0)

        # axes[0].imshow(spectrum_img, cmap=self.colormap_)
        # axes[1].imshow(audio_part_img, cmap=self.colormap_)

        # texture_file = f"{self.filename_}_texture_frame_{current_frame}_temp.png"
        # plt.savefig(texture_file, dpi=fig.dpi)
        # texture = QImage(texture_file).mirrored(False, False)
        # Path(texture_file).unlink()
        # plt.close(fig)
        return texture

    def set_position(self, position: float):
        self.texture_.setData(self.prepare_texture(position))

    def get_thumbnail(self) -> QPixmap:
        pixmap = None
        if self.thumbnail_ is not None:
            pixmap = self.thumbnail_
        else:
            pixmap = QPixmap(100, 100)
            pixmap.fill(Qt.black)
        return pixmap

    def can_be_binded(self):
        return True

