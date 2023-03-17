## Improvements

### Tokenizerrs/Models

- Vocabulary as a dictionary (now is a str)

### Converters
- [ ] Add MusicXML
- [ ] Add ABC notation.
- [ ] JSON to musicaiz objects.

### Plotters
- [ ] Adjust plotters. Plot in secs or ticks and be careful with tick labels in plots that have too much data,
numbers can overlap and the plot won't be clean.

### Harmony
- [ ] Measure just the correct interval (and not all the possible intervals based on the pitch) if note name is known (now it measures all the possible intervals given pitch, but if we do know the note name the interval is just one).
- [ ] Support key changes in middle of a piece when loading with ``loaders.Musa`` object.
- [ ] Initialize note names correctly if key or tonality is known (know the note name initialization is arbitrary, can be the enharmonic or not)

### Features
- [ ] Function to compute: Polyphonic rate
- [ ] Function to compute: Polyphony

### Tokenizers
- [ ] MusicTransformer
- [ ] Octuple
- [ ] Compound Word

### Synthesis
- [ ] Add function to synthesize a ``loaders.Musa`` object (can be inherited from ``pretty_midi``).

### Other TODOs
- [ ] Harmony: cadences
- [ ] Rhythm: sincopation
- [ ] Synzesize notes to be able to play chords, intervals, scales...(this might end being a plugin for composition assistance).