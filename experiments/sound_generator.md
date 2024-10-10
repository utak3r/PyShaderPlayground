Let's collect some sound material for verifying the corectness of the whole sound visualization system :)

Generate simple sine wave of a given frequency, sample rate and length:


```python
import numpy as np
from scipy.io import wavfile

sampleRate = 44100
frequency = 440
length = 10

t = np.linspace(0, length, sampleRate * length)
y = np.sin(frequency * 2 * np.pi * t)
    
m = np.max(np.abs(y))
maxint16 = np.iinfo(np.int16).max
y = maxint16 * y / m
y = y.astype(np.int16) 

wavfile.write(f'sine-freq{frequency}Hz-sr{sampleRate}-len{length}.wav', sampleRate, y)
```

    m 0.9999999999936564
    maxint16 32767
    

Let's try some sweep:


```python
import numpy as np
from scipy import signal
from scipy.io import wavfile

sampleRate = 44100
frequency_start = 100
frequency_end = 800
length = 10

t = np.linspace(0, length, sampleRate * length)
y = signal.chirp(t, f0=frequency_start, f1=frequency_end, t1=length)

m = np.max(np.abs(y))
maxint16 = np.iinfo(np.int16).max
y = maxint16 * y / m
y = y.astype(np.int16) 

wavfile.write(f'sinesweep-freq{frequency_start}Hz-freq{frequency_end}-sr{sampleRate}-len{length}.wav', sampleRate, y)
```
