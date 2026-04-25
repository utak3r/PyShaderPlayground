from PySide6.QtGui import QImage, QPixmap
from PySide6.QtOpenGL import QOpenGLTexture
from PySide6.QtCore import Qt
from enum import Enum
from pathlib import Path
from scipy.fftpack import fft, ifft
from scipy.signal.windows import hann
import numpy as np
import librosa
import simpleaudio
from PIL import Image
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import FigureCanvas

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

    def bind(self, unit: int = 0):
        self.get_texture().bind(unit)
    
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
        self.filter_minification_ = TextureFilter.FILTER_LINEAR
        self.filter_magnification_ = TextureFilter.FILTER_LINEAR
        self.create_texture(filename)

    def get_audio_duration(self):
        return self.duration_

    def get_audio_duration_ceiling(self):
        return np.ceil(self.duration_)

    @classmethod
    def get_audio_part(cls, audio, time_start=0.0, sample_rate=44100, num_samples=512):
        sample_start = int(time_start * sample_rate)
        sample_end = sample_start + num_samples
        
        # Handle padding if we reach the end of the audio
        if sample_end > len(audio):
            audio_part = audio[sample_start:]
            audio_part = np.pad(audio_part, (0, num_samples - len(audio_part)), 'constant')
        else:
            audio_part = audio[sample_start:sample_end]
            
        return audio_part

    @classmethod
    def wave_to_pixmap(cls, signal, sr, width, height) -> Image:
        fig = plt.figure(figsize=(width/100.0, height/100.0), dpi=100)
        canvas = FigureCanvas(fig)
        ax = plt.axes()
        ax.set_axis_off()
        ax.margins(0)
        ax.plot(np.arange(signal.size) / sr, signal)
        fig.tight_layout()
        canvas.draw()
        buffer_rgba = canvas.buffer_rgba()
        pixmap = QPixmap(
            QImage(
                buffer_rgba, 
                buffer_rgba.shape[1], 
                buffer_rgba.shape[0], 
                QImage.Format.Format_RGBA8888
                ).scaled(width, height, Qt.IgnoreAspectRatio))
        plt.close(fig)
        return pixmap

    def create_texture(self, filename: str):
        super().create_texture()
        self.filename_ = filename
        
        self.texture_.setSize(512, 2)
        self.texture_.setFormat(QOpenGLTexture.TextureFormat.R8_UNorm)
        self.texture_.setMipLevels(1)
        self.texture_.allocateStorage()

        INPUT_SAMPLE_RATE = None
        self.audio_, self.sample_rate_ = librosa.load(self.filename_, mono=True, sr=INPUT_SAMPLE_RATE)
        num_samples = self.audio_.shape[0]
        self.duration_ = num_samples / self.sample_rate_
        self.max_sample_value_ = np.max(self.audio_)

        self.thumbnail_ = InputTextureSound.wave_to_pixmap(self.audio_, self.sample_rate_, 100, 100)
        
        self.texture_.setData(QOpenGLTexture.PixelFormat.Red, 
                             QOpenGLTexture.PixelType.UInt8, 
                             self.prepare_texture(0.0))

        # # Playable audio (Stereo if available, native SR for best quality)
        # self.audio_playable, native_sr = librosa.load(self.filename_, sr=None, mono=False)
        
        # # Normalize to int16 range to avoid clipping and ensure audible volume
        # max_val = np.max(np.abs(self.audio_playable))
        # if max_val > 0:
        #     self.audio_playable = self.audio_playable / max_val * 32767
        
        # self.audio_playable = self.audio_playable.astype(np.int16)
        
        # # Simpleaudio expects (samples, channels) for multi-channel
        # if self.audio_playable.ndim > 1:
        #     # librosa returns (channels, samples), simpleaudio wants (samples, channels)
        #     self.audio_playable = np.ascontiguousarray(self.audio_playable.T)
        #     num_channels = self.audio_playable.shape[1]
        # else:
        #     num_channels = 1
            
        # self.audio_play_object = simpleaudio.WaveObject(self.audio_playable, 
        #                                                 num_channels=num_channels, 
        #                                                 bytes_per_sample=2, 
        #                                                 sample_rate=int(native_sr))
        # self.audio_play_playback = None

    def play_audio(self):
        return
        # if self.audio_play_playback:
        #     if self.audio_play_playback.is_playing():
        #         self.audio_play_playback.stop()
        #         self.audio_play_playback.wait_done()
        #         self.audio_play_playback = None
        #     else:
        #         self.audio_play_playback = self.audio_play_object.play()
        # else:
        #     self.audio_play_playback = self.audio_play_object.play()

    @classmethod
    def calculate_spectrum(cls, signal):
        # Windowed FFT (Hann filter) of 2048 samples to get 1024 bins,
        # Take first 512 bins (0 to 11025 Hz).
        window = np.hanning(len(signal))
        windowed_signal = signal * window
        fft_res = np.fft.rfft(windowed_signal)
        
        magnitude = np.abs(fft_res)[:512]

        # Flatten and rescale into 0..1 range
        magnitude = np.log10(magnitude + 1.0)
        magrange = np.max(magnitude) - np.min(magnitude)
        magnitude -= np.min(magnitude)
        magnitude /= magrange
                
        return magnitude[:512]

    def prepare_texture(self, position: float):
        wave_samples = InputTextureSound.get_audio_part(self.audio_, position, self.sample_rate_, 512)
        fft_samples = InputTextureSound.get_audio_part(self.audio_, position, self.sample_rate_, 2048)
        
        spectrum = InputTextureSound.calculate_spectrum(fft_samples)
        
        # Normalize Waveform: -1..1 -> 0..1 (0.5 is silence)
        wave_norm = (wave_samples + 1.0) / 2.0
        # wave_norm = wave_samples
        wave_norm = np.clip(wave_norm, 0.0, 1.0)
        
        # Normalize Spectrum: 0..? -> 0..1
        spec_norm = np.clip(spectrum, 0.0, 1.0)
        
        # Build texture
        data = np.zeros((2, 512), dtype=np.uint8)
        data[0, :] = (spec_norm * 255).astype(np.uint8)
        data[1, :] = (wave_norm * 255).astype(np.uint8)
        
        return data.tobytes()

    def set_position(self, position: float):
        if DEBUG_USE_SET_AUDIO_POSITION:
            position = DEBUG_AUDIO_POSITION
        if position <= self.get_audio_duration() and position >= 0.0:
            if position != self.current_position_:
                if self.is_texture_created():
                    self.texture_.setData(QOpenGLTexture.PixelFormat.Red, 
                                         QOpenGLTexture.PixelType.UInt8, 
                                         self.prepare_texture(position))
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
