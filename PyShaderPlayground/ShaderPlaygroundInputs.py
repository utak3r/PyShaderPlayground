from PySide6.QtGui import QImage, QPixmap
from PySide6.QtOpenGL import QOpenGLTexture
from PySide6.QtCore import Qt
from enum import Enum
from pathlib import Path
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq
from scipy.fft import rfft
from scipy.signal.windows import hann
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors as clrs
from skimage.transform import resize
from skimage import exposure
import skimage.io
import librosa
from PIL import Image

DEBUG_USE_SET_AUDIO_POSITION = False
DEBUG_AUDIO_POSITION = 14.0

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
        self.current_position_ = 0

    def get_texture(self) -> QOpenGLTexture:
        return self.texture_

    def create_texture(self):
        self.texture_ = QOpenGLTexture(QOpenGLTexture.Target2D)
        self.texture_.setMinificationFilter(self.filter_minification_.value)
        self.texture_.setMagnificationFilter(self.filter_magnification_.value)
        self.texture_.setWrapMode(self.wrapping_.value)
    
    def is_texture_created(self) -> bool:
        created = False
        if self.texture_:
            if self.texture_.isCreated():
                created = True
        return created
    
    def destroy_texture(self):
        if self.texture_:
            if self.texture_.isCreated():
                self.texture_.destroy()

    def get_thumbnail(self) -> QPixmap:
        pixmap = QPixmap(100, 100)
        pixmap.fill(Qt.black)
        return pixmap

    def can_be_binded(self):
        return False

    def bind(self):
        self.get_texture().bind()
    
    def release(self):
        self.get_texture().release()
    
    def is_bound(self) -> bool:
        return self.get_texture().isBound()

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
        self.duration_ = 0
        self.max_sample_value_ = 0
        self.current_frame_ = 0
        self.thumbnail_ = None
        #self.colormap_ = InputTextureSound.create_color_map()
        self.create_texture(filename)

    @staticmethod
    def array_to_red_image(array) -> Image:
        """Makes an image from NDArray. Array values are transferred into R channel of RGB."""
        img = None
        # get rid of imaginary data
        array = array.real
        # get absolute values
        array = abs(array)
        # get max value
        max = np.max(array)
        # normalize to 0.0 - 1.0 range
        arrayuint8 = array.astype(np.float64) / max
        # make it uint8 data
        arrayuint8 = 255 * arrayuint8
        # grey image from array
        img = Image.fromarray(arrayuint8.astype(np.uint8), mode='L')
        # empty grey image
        zero = np.zeros(array.shape, dtype=np.uint8)
        img_zero = Image.fromarray(zero, mode='L')
        # merge it. Real image goes to R channel, while G and B channels filled with zeroes
        img = Image.merge(mode='RGB', bands=(img, img_zero, img_zero))
        return img

    @staticmethod
    def fft_filtered(signal):
        """Calculates FFT of the real part of the signal and applies a Hann window"""
        N = signal.shape[0]
        w = hann(N) # Hann window
        c_w = abs(sum(w))
        fft = rfft(signal * w) / c_w 
        return fft


    @staticmethod
    def fft_db(input):
        """Calculates FFT of the real part of the signal and scales amplitudes to dB"""
        fft = np.abs(np.fft.rfft(input))
        fftdb = 20 * np.log10(fft)
        return fftdb

    
    @staticmethod
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled


    def create_texture(self, filename: str):
        super().create_texture()
        self.filename_ = filename
        # We're NOT resampling the source, leaving the original sample reate.
        # Also, we're loading it as a mono sound.
        INPUT_SAMPLE_RATE = None
        self.audio_, self.sample_rate_ = librosa.load(self.filename_, mono=True, sr=INPUT_SAMPLE_RATE)
        # if self.audio_.ndim > 1:
        #     self.audio_ = np.mean(self.audio_, axis=1) # we want mono!
        num_samples = self.audio_.shape[0]
        self.duration_ = num_samples / self.sample_rate_
        self.max_sample_value_ = np.max(self.audio_)

        audio_wave_img = self.array_to_red_image(self.audio_)
        audio_wave_img = audio_wave_img.rotate(-90, expand=True)
        audio_wave_img = audio_wave_img.resize((100,100))
        img = QImage(audio_wave_img.tobytes(), 100, 100, 100*3, QImage.Format_RGB888)
        self.thumbnail_ = QPixmap(img)

        self.texture_.setData(self.prepare_texture(0.0))

    def prepare_texture(self, position: float):
        current_frame = int(position*self.framerate_)
        samples_per_frame = int(self.sample_rate_ / self.framerate_)
        sample_start = int(position * self.sample_rate_)
        sample_end = sample_start + samples_per_frame
        N = sample_end - sample_start
        T = 1.0 / self.sample_rate_
        audio_part = self.audio_[sample_start:sample_end]
        self.current_frame_ = current_frame

        audio_part_img = exposure.rescale_intensity(audio_part, out_range=(-1.0, 1.0))
        audio_wave_img = self.array_to_red_image(audio_part_img)
        audio_wave_img = audio_wave_img.rotate(90, expand=True)
        audio_wave_img = audio_wave_img.resize((512,1))

        fft = self.fft_filtered(audio_part)
        reduced_N = int(fft.shape[0]/2)
        reduced_fft = fft[0:reduced_N]
        fft_img = self.array_to_red_image(reduced_fft)
        fft_img = fft_img.rotate(90, expand=True)
        fft_img = fft_img.resize((512,1))
        spectrogram_image = fft_img.resize((512,1))

        width_spec, height_spec = spectrogram_image.size
        width_wave, height_wave = audio_wave_img.size

        final_img = Image.new('RGB', size=(width_spec,height_spec+height_wave))
        final_img.paste(spectrogram_image, (0, 0))
        final_img.paste(audio_wave_img, (0, height_spec))
        final_width, final_height = final_img.size

        texture = QImage(final_img.tobytes(), final_width, final_height, final_width*3, QImage.Format_RGB888)
        #texture.save('final_texture.jpg')
        return texture

    def set_position(self, position: float):
        if DEBUG_USE_SET_AUDIO_POSITION:
            position = DEBUG_AUDIO_POSITION
        if position != self.current_position_:
            if self.is_texture_created():
                super().destroy_texture()
                super().create_texture()
            self.texture_.setData(self.prepare_texture(position))
            self.current_position_ = position

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

