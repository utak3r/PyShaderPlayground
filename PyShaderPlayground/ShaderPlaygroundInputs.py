from PySide6.QtGui import QImage, QPixmap
from PySide6.QtOpenGL import QOpenGLTexture
from PySide6.QtCore import Qt
from enum import Enum
from pathlib import Path
from scipy.fftpack import fft, ifft
from scipy.signal.windows import hann
import numpy as np
import librosa
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
    
    def get_texture_filename(self) -> str:
        return self.filename_

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

    def get_audio_duration(self) -> bool:
        return self.duration_

    @classmethod
    def get_audio_part(cls, audio, time_start=0.0, sample_rate=44100, frame_rate=30, nframes = 1):
        samples_per_frame = int(sample_rate / frame_rate)
        sample_start = int(time_start * sample_rate)
        sample_end = sample_start + (nframes * samples_per_frame)
        N = sample_end - sample_start
        T = 1.0 / sample_rate
        t = np.linspace(0, nframes * (1.0 / frame_rate), int(sample_rate * nframes * (1.0 / frame_rate)))
        audio_part = audio[sample_start:sample_end]
        return (audio_part, N, T, t)

    @classmethod
    def array_to_red_image(cls, array) -> Image:
        img = None
        # get absolute values
        array = abs(array)
        # get max value
        max = np.max(array)
        # normalize to 0.0 - 1.0 range
        if max > 0:
            arrayuint8 = array.astype(np.float64) / max
        else:
            arrayuint8 = array.astype(np.float64)
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
    
    @classmethod
    def array_to_rgb_image(cls, array) -> Image:
        img = None
        return img

    @classmethod
    def merge_images(cls, img1: Image, img2: Image):
        img = None
        final_width = np.max([img1.size[0], img2.size[0]])
        final_height = img1.size[1] + img2.size[1]
        img = Image.new('RGB', size=(final_width, final_height))
        img.paste(img1, (0, 0))
        img.paste(img2, (0, img1.size[1]))
        return img
    
    @classmethod
    def transform_image(cls, img: Image, rotation=0.0, width=1024, height=1024):
        img_out = img.rotate(rotation, expand=True)
        img_out = img_out.resize((width, height))
        return img_out

    @classmethod
    def wave_to_img(cls, signal, sr, width, height) -> Image:
        fig = plt.figure(figsize=(width/100.0, height/100.0), dpi=100)
        canvas = FigureCanvas(fig)
        ax = plt.axes()
        ax.set_axis_off()
        ax.margins(0)
        ax.plot(np.arange(signal.size) / sr, signal)
        fig.tight_layout()
        canvas.draw()
        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        img = Image.fromarray(img, mode='RGB')
        img = img.resize((width, height), resample=Image.BILINEAR)
        plt.close(fig)
        return img

    @classmethod
    def image_to_qimage(cls, image, width, height) -> QImage:
        return QImage(image.tobytes(), width, height, width*3, QImage.Format_RGB888)

    def create_texture(self, filename: str):
        super().create_texture()
        self.filename_ = filename
        # We're NOT resampling the source, leaving the original sample reate.
        # Also, we're loading it as a mono sound.
        INPUT_SAMPLE_RATE = None
        self.audio_, self.sample_rate_ = librosa.load(self.filename_, mono=True, sr=INPUT_SAMPLE_RATE)
        num_samples = self.audio_.shape[0]
        self.duration_ = num_samples / self.sample_rate_
        self.max_sample_value_ = np.max(self.audio_)

        self.thumbnail_ = QPixmap(InputTextureSound.image_to_qimage(InputTextureSound.wave_to_img(self.audio_, self.sample_rate_, 100, 100), 100, 100))
        self.texture_.setData(self.prepare_texture(0.0))

    @classmethod
    def calculate_magnitude_db(cls, value):
        '''Calculate magnitude of a complex value and scale it in dB.'''
        '''Do some simplifying things, like zeroing some values before log10.'''
        '''Also note, we're not multiplying it by 20, as it will be rescaled afterwards anyway,'''
        '''so the magnitude values are 1/20 dB in fact.'''
        result = 0
        magnitude  = np.sqrt(np.pow(value.real, 2) + np.pow(value.imag, 2))
        if magnitude < 1.0:
            result = 0.0
        else:
            result = np.log10(magnitude)
        return result

    @classmethod
    def calculate_spectrum(cls, signal, min_value, max_value, N):
        signal_fft = np.fft.rfft(signal)
        N_spectrum = int(signal_fft.size/2)
        spectrum = np.linspace(start=min_value, stop=max_value, num=N)
        
        for i in range(0, N_spectrum):
            magnitude = InputTextureSound.calculate_magnitude_db(signal_fft[i])
            spectrum[i] = magnitude
        return spectrum

    def prepare_texture(self, position: float):
        audio_part, N, T, t = InputTextureSound.get_audio_part(self.audio_, time_start=position, sample_rate=self.sample_rate_, frame_rate=self.framerate_, nframes=1)
        self.current_frame_ = int(position*self.framerate_)

        audio_wave_img = InputTextureSound.array_to_red_image(audio_part)
        audio_wave_img = InputTextureSound.transform_image(audio_wave_img, 90, 512, 1)

        audio_spectrum = InputTextureSound.calculate_spectrum(audio_part, 0.0, (self.sample_rate_ / 4.0), int(N/4))
        spectrogram_image = InputTextureSound.array_to_red_image(audio_spectrum)
        spectrogram_image = InputTextureSound.transform_image(spectrogram_image, 90, 512, 1)

        final_img = InputTextureSound.merge_images(spectrogram_image, audio_wave_img)
        final_width, final_height = final_img.size
        texture = QImage(final_img.tobytes(), final_width, final_height, final_width*3, QImage.Format_RGB888)
        #texture.save('final_texture.jpg')
        return texture

    def set_position(self, position: float):
        if DEBUG_USE_SET_AUDIO_POSITION:
            position = DEBUG_AUDIO_POSITION
        if position <= self.get_audio_duration():
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

