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
        self.length_ = 0
        self.max_sample_value_ = 0
        self.current_frame_ = 0
        self.thumbnail_ = None
        #self.colormap_ = InputTextureSound.create_color_map()
        self.create_texture(filename)

    @staticmethod
    def array_to_red_image(array) -> Image:
        """Makes an image from NDArray. Array values are transferred into R channel of RGB."""
        img = None
        # grey image from array
        img = Image.fromarray(array, mode='L')
        # empty grey image of the same size
        zero = np.zeros(array.shape, dtype=np.uint8)
        img_zero = Image.fromarray(zero, mode='L')
        # merge it. Real image goes to R channel, while G and B channels filled with zeroes
        img = Image.merge(mode='RGB', bands=(img, img_zero, img_zero))
        return img
    
    @staticmethod
    def scale_minmax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled


    def create_texture(self, filename: str):
        super().create_texture()
        self.filename_ = filename
        # we're converting any input into predefined sample rate
        INPUT_SAMPLE_RATE = 22500
        self.audio_, self.sample_rate_ = librosa.load(self.filename_, sr=INPUT_SAMPLE_RATE)
        if self.audio_.ndim > 1:
            self.audio_ = np.mean(self.audio_, axis=1) # we want mono!
        num_samples = self.audio_.shape[0]
        self.length_ = num_samples / self.sample_rate_
        self.max_sample_value_ = np.max(self.audio_)

        audio_part_img = exposure.rescale_intensity(self.audio_, out_range=(-1.0, 1.0))
        audio_wave_img = self.array_to_red_image(audio_part_img)
        audio_wave_img = audio_wave_img.rotate(-90, expand=True)
        audio_wave_img = audio_wave_img.resize((100,100))
        img = QImage(audio_wave_img.tobytes(), 100, 100, 100*3, QImage.Format_RGB888)
        self.thumbnail_ = QPixmap(img)

        # waveform = librosa.display.waveshow(self.audio_, sr=self.sample_rate_)
        # img = self.scale_minmax(waveform, 0, 255).astype(np.uint8)
        # img = img.resize((100,100))
        # img2 = QImage(img.data, 100, 100, QImage.Format_Indexed8)
        # self.thumbnail_ = QPixmap(img2)

        # fig=plt.figure(figsize=(1.0, 1.0), dpi=100)
        # canvas = FigureCanvas(fig)
        # ax = plt.axes()
        # ax.set_axis_off()
        # ax.margins(0)
        # ax.plot(np.arange(num_samples) / self.sample_rate_, self.audio_)
        # fig.tight_layout()
        # thumb_file = f"{self.filename_}_thumbnail_temp.png"

        # canvas.draw()
        # img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(100, 100, 3)
        # img2 = QImage(img.data, 100, 100, QImage.Format_Indexed8)

        #plt.savefig(thumb_file, dpi=fig.dpi)
        #plt.close(fig)
        #self.thumbnail_ = QPixmap(QImage(thumb_file))
        #self.thumbnail_ = QPixmap(img2)
        #Path(thumb_file).unlink()
        self.texture_.setData(self.prepare_texture(0.0))

    def prepare_texture(self, position: float):
        current_frame = int(position*self.framerate_)
        sample_start = int(current_frame / self.framerate_ * self.sample_rate_)
        sample_end = int((current_frame + 1) / self.framerate_ * self.sample_rate_)
        N = sample_end - sample_start
        T = 1.0 / self.sample_rate_
        audio_part = self.audio_[sample_start:sample_end]

        audio_part_img = exposure.rescale_intensity(audio_part, out_range=(-1.0, 1.0))
        audio_wave_img = self.array_to_red_image(audio_part_img)
        audio_wave_img = audio_wave_img.rotate(-90, expand=True)
        audio_wave_img = audio_wave_img.resize((1024,512))

        n_fft = int(N)
        mel_spect = librosa.feature.melspectrogram(y=audio_part, sr=self.sample_rate_, n_fft=n_fft, hop_length=n_fft)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        melspect_img = self.array_to_red_image(mel_spect)
        melspect_img = melspect_img.crop((0, 0, 1, melspect_img.size[1]))
        melspect_img = melspect_img.rotate(-90, expand=True)
        spectrogram_image = melspect_img.resize((1024,512))

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
        if position is not self.current_position_:
            if self.is_texture_created():
                super().destroy_texture()
                super().create_texture()
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

