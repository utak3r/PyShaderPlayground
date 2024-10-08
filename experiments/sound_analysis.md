Our main goal is to be able to represent loaded music. We will use FFT for that.
What are the needs?
We're creating a video, let's say, 5 mins long, of 60fps. That means, we have 18.000 video frames in total.
If the incoming audio is of 44.100 sample rate, for each video frame we have 735 samples, so that would be our window for FFT analysis.

But, there's another thing to consider. The resulting texture.
Should it be updated each frame and contain only that window? Time consuming, we could fall off off being real time.
Or maybe we should prepare one texture of the whole song, where on Y axis it'd have FFT for a single frame, and frames would be put on X axis? Tat would be best, BUT - in case of our sampe 5mins long video it'd mean 18K length, that's not an option to load up as a texture into OpenGL...


```python
from scipy.fftpack import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import colors as clrs
from skimage.transform import resize
from skimage import exposure
from scipy.signal.windows import gaussian
import librosa
from PIL import Image
```

Ok, let's load some music


```python
filename = 'test_sound_01.mp3'
audio, sample_rate = librosa.load(filename, sr=44100)
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
num_samples = audio.shape[0]
length = num_samples / sample_rate
max_sample_value = np.max(audio)

print(f'Audio sample rate: {sample_rate}, number of samples: {num_samples}. Length: {length}, max sample value is: {max_sample_value}')
```

    Audio sample rate: 44100, number of samples: 2526401. Length: 57.288004535147394, max sample value is: 0.8663547039031982
    

Let's draw simple waveform and frequency spectrum analysis:


```python
plt.plot(audio);
plt.title('Signal');
plt.xlabel('Time (samples)');
plt.ylabel('Amplitude');
plt.show()

n_fft = 2048
ft = np.abs(librosa.stft(audio[:n_fft], hop_length = n_fft+1))
plt.plot(ft);
plt.title('Spectrum');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');
plt.show()
```


    
![png](sound_analysis_files/sound_analysis_5_0.png)
    



    
![png](sound_analysis_files/sound_analysis_5_1.png)
    


Now, let's see a full spectrogram of a whole loaded music. It's scaled in dB.
Note: FFT (and short time FFT, too) gives both positive and negative results. 
The sign here is unrelevant for us, we care only about the absolute value.


```python
spec = np.abs(librosa.stft(audio, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');
plt.title('Spectrogram');
plt.show()
```


    
![png](sound_analysis_files/sound_analysis_7_0.png)
    


Now...
as humans do not perceive frequencies on a linear scale (we're detecting lower freqs much better than higher freqs), 
we will use what is called a mel scale (proposed by Stevens, Volkmann, and Newmann back in 1937).


```python
mel_spect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=2048, hop_length=1024)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
librosa.display.specshow(mel_spect, y_axis='mel', fmax=20000, x_axis='time');
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');
plt.show()
```


    
![png](sound_analysis_files/sound_analysis_9_0.png)
    


Now we prepare the color map. It's using only a red color, so later,
in our visualizations of audio signal, we will be checking only one channel of a texture.


```python
# prepare a red only colormap
reds = plt.get_cmap('Reds', 256)
black_reds = reds(np.linspace(0, 1, 256))
for i in range(256):
    black_reds[i:i+1, :] = np.array([i/256, 0, 0, 1])
map_black_reds = clrs.ListedColormap(black_reds)
colormap = map_black_reds
```

Let's start working on our texture. 
We want it to be made of two parts: one for showing up the waveform of a short time window,
and the another for showing up a mel spectrogram of the same window.


```python
frame_rate = 30.0 # framerate of a video
current_time = 14.0 # in seconds
current_frame = int(current_time * frame_rate)
number_of_frames = 1
sample_start = int((current_frame) / frame_rate * sample_rate)
sample_end = int((current_frame + number_of_frames) / frame_rate * sample_rate)
audio_part = audio[sample_start:sample_end]
```

The waveform part is easy and straightforward.


```python
def array_to_red_image(array):
    img = None
    # grey image from array
    img = Image.fromarray(array, mode='L')
    # empty grey image
    zero = np.zeros(array.shape, dtype=np.uint8)
    img_zero = Image.fromarray(zero, mode='L')
    # merge it. Real image goes to R channel, while G and B channels filled with zeroes
    img = Image.merge(mode='RGB', bands=(img, img_zero, img_zero))
    return img
```


```python
audio_part_img = exposure.rescale_intensity(audio_part, out_range=(-1.0, 1.0))
print(f'Waveform size: {audio_part_img.size}')
# convert to image
audio_wave_img = array_to_red_image(audio_part_img)
# rotate and resize
audio_wave_img = audio_wave_img.rotate(-90, expand=True)
audio_wave_img = audio_wave_img.resize((1024,512))
#audio_wave_img.save('audiowave_array.png')
display(audio_wave_img)
```

    Waveform size: 1470
    


    
![png](sound_analysis_files/sound_analysis_16_1.png)
    


For a mel spectrogram, we will do some cropping, rotating etc. ;)


```python
number_of_samples = sample_end-sample_start
n_fft = int(number_of_samples)
print(f'number of samples: {number_of_samples}, n_FFT: {n_fft}')
mel_spect = librosa.feature.melspectrogram(y=audio_part, sr=sample_rate, n_fft=n_fft, hop_length=n_fft)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
melspect_img = array_to_red_image(mel_spect)

# take first column
melspect_img = melspect_img.crop((0, 0, 1, melspect_img.size[1]))
# rotate and resize
melspect_img = melspect_img.rotate(-90, expand=True)
spectrogram_image = melspect_img.resize((1024,512))

display(spectrogram_image)
#spectrogram_image.save('spectrogram_array.png')
```

    number of samples: 1470, n_FFT: 1470
    


    
![png](sound_analysis_files/sound_analysis_18_1.png)
    


So, we have all required parts to construct a final texture.
Note: t was calculated for a current video frame, so it would have to be
recalculated on each frame!

We take both parts and concatenate into one square texture.


```python
width_spec, height_spec = spectrogram_image.size
width_wave, height_wave = audio_wave_img.size
print(f'Img1 size: {width_spec}:{height_spec}')
print(f'Img2 size: {width_wave}:{height_wave}')

final_img = Image.new('RGB', size=(width_spec,height_spec+height_wave))
final_img.paste(spectrogram_image, (0, 0))
final_img.paste(audio_wave_img, (0, height_spec))
final_img.save('final_texture.png')
print(f'Final image size: {final_img.size}')
display(final_img)
```

    Img1 size: 1024:512
    Img2 size: 1024:512
    Final image size: (1024, 1024)
    


    
![png](sound_analysis_files/sound_analysis_20_1.png)
    

