# My Modules

These are the modules that generate a sound stimulus.

## Requires

- numpy
- scipy
- pyloudnorm

## Modules

### akeroyd.py

----------

#### Methods

##### Generate()

Gerenate a Binaural Beat constructed from bandpass filtered noise using Akeroyd's method.

###### Parameters

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

###### Returns

Output a 32-bit float wav file.

##### GenerateInitIpd()

Generate a Binaural Beat constructed from bandpass filtered noise using Akeroyd's method with an initial interaural phase difference.

###### Parameters

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
- init_ipd : float
  - Initial IPD in degree.
- file_name : str
  - Output file name. (optional)

###### Returns

Output a 32-bit float wav file.

### phasewarp.py

----------

### oscor.py

### phase_delay.py

### pd_shift.py

### modulation.py
