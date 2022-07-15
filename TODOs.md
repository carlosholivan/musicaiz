### TODOs
- [ ] datasets submodule to download and process common datasets for music generation

## Improvements
### Converters
- [ ] Add protobufs.
- [ ] Add MusicXML
- [ ] Add ABC notation.

### Rhythm
- [ ] Support time signature changes in middle of a piece when loading with ``loaders.Musa`` object.
- [ ] Support tempo or bpm changes in middle of a piece when loading with ``loaders.Musa`` object.

### Plotters
- [ ] Adjust plotters. Plot in secs or ticks and be careful with tick labels in plots that have too much data,
numbers can overlap and the plot won't be clean.
- [ ] Plot all tracks in the same pianoroll.

### Harmony
- [ ] Measure just the correct interval (and not all the possible intervals based on the pitch) if note name is known (now it measures all the possible intervals given pitch, but if we do know the note name the interval is just one)

### Features
- [ ] Implement paper for rhythm patterns detection
- [ ] Initialize note names correctly if tonality is known (know the note name initialization is arbitrary, can be the enharmonic or not)

### Tokenizers
- [ ] Add more encodings (MusicBERT...)

### Synthesis
- [ ] Add function to synthesize a ``loaders.Musa`` object (can be inherited from ``pretty_midi``).

### Other TODOs
- [ ] Harmony: cadences
- [ ] Rhythm: sincopation
- [ ] Synzesize notes to be able to play chords, intervals, scales...(this might end being a plugin for composition assistance).