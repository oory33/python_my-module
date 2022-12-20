# My Modules

These are the modules that generate a sound stimulus.

## Requires

- numpy
- scipy
- pyloudnorm

## Modules

### akeroyd.py

----------

Gerenate a Binaural Beat constructed ftom bandpass filtered noise using Akeroyd's method.

#### Parameters

- srate : int
  - Sampling rate.
- shift : int
  - Shift frequency in Hz.
- duration : int
  - Total duration in seconds.
- bwd : int
  - Bandwidth in Hz.
- centre : int
  - Centre frequency of bandpass filter in Hz.
- init_direction : str
  - Initial direction of shift. Either "left" or "right".

#### Returns

Output a 32-bit float wav file.

### phasewarp.py

----------

### oscor.py

### phase_delay.py

### pd_shift.py

### modulation.py
