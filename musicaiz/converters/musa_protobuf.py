from musicaiz.converters.protobuf import musicaiz_pb2
from musicaiz.structure import Note, Instrument
from musicaiz.loaders import Musa


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
    time_sig = proto.time_signatures.add()
    time_sig.num = musa_obj.time_sig.num
    time_sig.denom = musa_obj.time_sig.denom

    # Other parameters (quantization, PPQ...)
    for instr in musa_obj.instruments:
        proto_instruments = proto.instruments.add()
        proto_instruments.program = instr.program
        proto_instruments.name = instr.name
        proto_instruments.family = instr.family if instr.family is not None else ""
        proto_instruments.is_drum = instr.is_drum

        if instr.bars is not None:
            if len(instr.bars) != 0:
            # loop in bars to add them to the protobuf
                for bar in instr.bars:
                    proto_bars = proto_instruments.bars.add()
                    proto_bars.bpm = bar.bpm
                    proto_bars.time_sig = bar.time_sig
                    proto_bars.resolution = bar.resolution
                    proto_bars.absolute_timing = bar.absolute_timing
                    proto_bars.note_density = bar.note_density
                    proto_bars.harmonic_density = bar.harmonic_density
                    proto_bars.start_ticks = bar.start_ticks
                    proto_bars.end_ticks = bar.end_ticks
                    proto_bars.start_sec = bar.start_sec
                    proto_bars.end_sec = bar.end_sec

                    # loop in notes to add them to the protobuf
                    for note in bar.notes:
                        proto_note = proto_bars.notes.add()
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
        else:
            if len(instr.notes) != 0:
                # loop in notes to add them to the protobuf
                for note in instr.notes:
                    proto_note = proto_instruments.notes.add()
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
    return proto


def proto_to_musa(protobuf) -> Musa:

    """
    Converts a protobuf to a Musa object.

    Returns
    -------
    
    midi: Musa
        The output Musa object.
    """
    
    midi = Musa()

    for i, instr in enumerate(protobuf.instruments):
        midi.instruments.append(
            Instrument(
                program=instr.program,
                name=instr.name,
                is_drum=instr.is_drum,
                general_midi=False,
            )
        )
        for note in instr.notes:
            # Initialize the note with musicaiz `Note` object
            midi.instruments[i].notes.append(
                Note(
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                    velocity=note.velocity,
                    bpm=protobuf.bpm,
                    resolution=protobuf.resolution,
                )
            )
    return midi