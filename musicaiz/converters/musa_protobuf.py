from musicaiz.converters.protobuf import musicaiz_pb2
from musicaiz.structure import (
    Note,
    Instrument,
    Bar,
)
from musicaiz.rhythm import (
    Beat,
    Subdivision,
)
from musicaiz import loaders


def musa_to_proto(musa_obj):
    """
    Converts the Musa class attributes to a protobuf.
    The protobuf follows the nomenclature and naming in musicaiz and the
    same logic as the Musa class. As an example, is we set Musa's arg
    `structure="bars"` the notes will be grouped inside the bars that'll be
    inside the instruments.

    Returns
    -------

    proto:
        The output protobuf.
    """
    proto = musicaiz_pb2.Musa()

    # Time signature data
    proto_time_signature_changes = proto.time_signature_changes.add()
    proto_time_signature_changes = musa_obj.time_signature_changes

    proto_subdivision_note = proto.subdivision_note.add()
    proto_subdivision_note = proto.subdivision_note.add()

    proto_file = proto.file.add()
    proto_file = musa_obj.file

    proto_total_bars = proto.total_bars.add()
    proto_total_bars = musa_obj.total_bars

    proto_tonality = proto.tonality.add()
    proto_tonality = musa_obj.tonality

    proto_resolution = proto.resolution.add()
    proto_resolution = musa_obj.resolution

    proto_is_quantized = proto.is_quantized.add()
    proto_is_quantized = musa_obj.is_quantized

    proto_quantizer_args = proto.quantizer_args.add()
    proto_quantizer_args = musa_obj.quantizer_args

    proto_absolute_timing = proto.absolute_timing.add()
    proto_absolute_timing = musa_obj.absolute_timing

    proto_cut_notes = proto.cut_notes.add()
    proto_cut_notes = musa_obj.cut_notes

    proto_tempo_changes = proto.tempo_changes.add()
    proto_tempo_changes = musa_obj.tempo_changes

    proto_instruments_progs = proto.instruments_progs.add()
    proto_instruments_progs = musa_obj.instruments_progs

    # Other parameters (quantization, PPQ...)
    for instr in musa_obj.instruments:
        proto_instruments = proto.instruments.add()
        proto_instruments.program = instr.program
        proto_instruments.name = instr.name
        proto_instruments.family = instr.family if instr.family is not None else ""
        proto_instruments.is_drum = instr.is_drum

    # loop in bars to add them to the protobuf
    for bar in musa_obj.bars:
        proto_bars = proto.bars.add()
        proto_bars.bpm = bar.bpm
        proto_bars.time_sig = bar.time_sig.time_sig
        proto_bars.resolution = bar.resolution
        proto_bars.absolute_timing = bar.absolute_timing
        proto_bars.note_density = bar.note_density
        proto_bars.harmonic_density = bar.harmonic_density
        proto_bars.start_ticks = bar.start_ticks
        proto_bars.end_ticks = bar.end_ticks
        proto_bars.start_sec = bar.start_sec
        proto_bars.end_sec = bar.end_sec

    # loop in notes to add them to the protobuf
    for note in musa_obj.notes:
        proto_note = proto.notes.add()
        proto_note.pitch = note.pitch
        proto_note.pitch_name = note.pitch_name
        proto_note.note_name = note.note_name
        proto_note.octave = note.octave
        proto_note.ligated = note.ligated
        proto_note.start_ticks = note.start_ticks
        proto_note.end_ticks = note.end_ticks
        proto_note.start_sec = note.start_sec
        proto_note.end_sec = note.end_sec
        proto_note.symbolic = note.symbolic
        proto_note.velocity = note.velocity
        proto_note.instrument_prog = note.instrument_prog
        proto_note.instrument_idx = note.instrument_idx
        proto_note.beat_idx = note.beat_idx
        proto_note.bar_idx = note.bar_idx
        proto_note.subbeat_idx = note.subbeat_idx

    for beat in musa_obj.beats:
        proto_beat = proto.beats.add()
        proto_beat.time_sig = beat.time_sig.time_sig
        proto_beat.bpm = beat.bpm
        proto_beat.global_idx = beat.global_idx
        proto_beat.bar_idx = beat.bar_idx
        proto_beat.start_sec = beat.start_sec
        proto_beat.end_sec = beat.end_sec
        proto_beat.start_ticks = beat.start_ticks
        proto_beat.end_ticks = beat.end_ticks

    for subbeat in musa_obj.subbeats:
        proto_subbeat = proto.subbeats.add()
        proto_subbeat.global_idx = subbeat.global_idx
        proto_subbeat.bar_idx = subbeat.bar_idx
        proto_subbeat.beat_idx = subbeat.beat_idx
        proto_subbeat.start_sec = subbeat.start_sec
        proto_subbeat.end_sec = subbeat.end_sec
        proto_subbeat.start_ticks = subbeat.start_ticks
        proto_subbeat.end_ticks = subbeat.end_ticks

    return proto


def proto_to_musa(protobuf): #-> loaders.Musa:

    """
    Converts a protobuf to a Musa object.
    The Musa object will be constructed
    with the same way data structure of the proto.
    Ex.: If the notes in the proto are organized inside bars, the
    Musa object will store the notes inside Bar Bar objects, as if
    we initialized the Musa object with `structure="bars"` argument.

    Returns
    -------

    midi: Musa
        The output Musa object.
    """

    midi = loaders.Musa(file=None)

    for _, instr in enumerate(protobuf.instruments):
        midi.instruments.append(
            Instrument(
                program=instr.program,
                name=instr.name,
                is_drum=instr.is_drum,
                general_midi=False,
            )
        )

    for note in midi.notes:
        # Initialize the note with musicaiz `Note` object
        midi.notes.append(
            Note(
                pitch=note.pitch,
                start=note.start_ticks,
                end=note.end_ticks,
                velocity=note.velocity,
                bpm=note.bpm,
                resolution=note.resolution,
                instrument_prog=note.instrument_prog,
                instrument_idx=note.instrument_idx
            )
        )

    for j, bar in enumerate(midi.bars):
        # Initialize the bar with musicaiz `Bar` object
        midi.bars.append(
            Bar(
                time_sig=bar.time_sig,
                bpm=bar.bpm,
                resolution=bar.resolution,
                absolute_timing=bar.absolute_timing,
            )
        )
        midi.bars[j].note_density = bar.note_density
        midi.bars[j].harmonic_density = bar.harmonic_density
        midi.bars[j].start_ticks = bar.start_ticks
        midi.bars[j].end_ticks = bar.end_ticks
        midi.bars[j].start_sec = bar.start_sec
        midi.bars[j].end_sec = bar.end_sec

    for beat in midi.beats:
        # Initialize the note with musicaiz `Note` object
        midi.beats.append(
            Beat(
                time_sig=beat.time_sig,
                bpm=beat.bpm,
                resolution=beat.resolution,
                global_idx=beat.global_idx,
                bar_idx=beat.bar_idx,
                start=beat.start_sec,
                end=beat.end_sec
            )
        )

    for subbeat in midi.subbeats:
        # Initialize the note with musicaiz `Note` object
        midi.notes.append(
            Subdivision(
                time_sig=subbeat.time_sig,
                bpm=subbeat.bpm,
                resolution=subbeat.resolution,
                global_idx=subbeat.global_idx,
                bar_idx=subbeat.bar_idx,
                start=subbeat.start_sec,
                end=subbeat.end_sec,
                beat_idx=subbeat.beat_idx,
            )
        )
    return midi
