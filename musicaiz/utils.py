from typing import List, Union, TextIO, Optional
from pathlib import Path


from musicaiz.structure import Note
from musicaiz.loaders import Musa
from musicaiz.rhythm import get_subdivisions, TimingConsts


def get_list_files_path(path: Union[Path, str]) -> List[str]:
    """Returns a list of all files in a directory as strings.
    """
    if isinstance(path, str):
        path = Path(path)
    files = list(path.rglob('*.mid'))
    files.extend(list(path.rglob('*.midi')))

    files_str = [str(f) for f in files]
    return files_str


def sort_notes(note_seq: List[Note]) -> List[Note]:
    """Sorts a list of notes by the start_ticks notes attribute."""
    note_seq.sort(key=lambda x: x.start_ticks, reverse=False)
    return note_seq


def group_notes_in_subdivisions_bars(musa_obj: Musa, subdiv: str) -> List[List[List[Note]]]:
    """This function groups notes in the selected subdivision.
    The result is a list which elements are lists that represent the bars,
    and inside them, lists that represent the notes in each subdivision.

    Parameters
    ----------
    musa_obj: Musa
        A Musa object initialized with the argument `structure="bars"`.

    Returns
    -------
    all_subdiv_notes: List[List[List[Note]]]
        A list of bars in which each element is a subdivision which is a List of notes that
        are in the subdivision.
        Ex.: For 4 bars at 4/4 bar and 8th note as subdivision, we'll have a list of 4 items
            beign each item a list of 8 elements which are the 8th notes that are in each 4/4 bar.
            Inside the subdivisions list, we'll find the notes that belong to the subdivision.
    """
    # Group notes in bars (no instruments)
    bars = Musa.group_instrument_bar_notes(musa_obj)
    # 1. Sort midi notes in all the bars
    sorted_bars = [sort_notes(b) for b in bars]
    # Retain the highest note at a time frame (1 16th note)
    grid = get_subdivisions(
        total_bars=len(sorted_bars),
        subdivision=subdiv,
        time_sig=musa_obj.time_sig.time_sig,
        bpm=musa_obj.bpm,
        resolution=musa_obj.resolution,
        absolute_timing=musa_obj.absolute_timing
    )
    step_ticks = grid[1]["ticks"]
    bar_grid = [g for g in grid if g["bar"] == 1]

    # Group notes in subdivisions
    all_subdiv_notes = []  # Lis with N items = N bars
    new_step_ticks = 0
    for b, bar in enumerate(sorted_bars):
        bar_notes = []  # List with N items = to N subdivisions per bar
        for s, subd in enumerate(bar_grid):
            subdiv_notes = []  # List of notes in the current subdivision
            start = new_step_ticks
            end = new_step_ticks + step_ticks
            for i, note in enumerate(bar):
                if note.start_ticks <= start and note.end_ticks >= end:
                    subdiv_notes.append(note)
            new_step_ticks += step_ticks
            bar_notes.append(subdiv_notes)
        all_subdiv_notes.append(bar_notes) 
    return all_subdiv_notes


def get_highest_subdivision_bars_notes(
    all_subdiv_notes: List[List[List[Note]]]
) -> List[List[Note]]:
    """Extracts the highest note in each subdivision.

    Parameters
    ----------
    all_subdiv_notes: List[List[List[Note]]]
        A list of bars in which each element is a subdivision which is a List of notes that
        are in the subdivision.

    Returns
    -------
    bar_highest_subdiv_notes: List[List[Note]]
        A list of bars in which each element in the bar carresponds to the note with the
        highest pitch in the subdivision.
    """
    # Retrieve the note with the highest pitch in each subdivision of every bar
    bar_highest_subdiv_notes = []
    for b, bar in enumerate(all_subdiv_notes):
        highest_subdiv_notes = []  # List of N items = each item is the highest note in the subdiv
        for s, subdiv in enumerate(bar):
            if len(subdiv) == 0:
                highest_subdiv_notes.append(Note(pitch=0, start=0, end=1, velocity=0))
                continue
            for n, note in enumerate(subdiv):
                prev_pitch = 0
                # If there are no notes in a bar, fill it with a note which pitch is 0
                if subdiv[n].pitch > prev_pitch:
                    highest_note = note
            highest_subdiv_notes.append(highest_note)
        bar_highest_subdiv_notes.append(highest_subdiv_notes)
    return bar_highest_subdiv_notes


def __initialization(
    file: Union[Musa, str, TextIO, Path],
    structure: str = "bars",
    quantize: bool = True,
    quantize_note: Optional[str] = "sixteenth",
    tonality: Optional[str] = None,
    time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
    bpm: int = TimingConsts.DEFAULT_BPM.value,
    resolution: int = TimingConsts.RESOLUTION.value,
    absolute_timing: bool = True
) -> Musa:
    """
    If both musa_obj and file are give, the class will be initialized with the given file
    not with the musa_obj.
    """
    if isinstance(file, Musa):
        musa_obj = file
    elif isinstance(file, str) or isinstance(file, Path) or isinstance(file, TextIO):
        musa_obj = Musa(
            file=file,
            structure=structure,
            quantize=quantize,
            quantize_note=quantize_note,
            tonality=tonality,
            time_sig=time_sig,
            bpm=bpm,
            resolution=resolution,
            absolute_timing=absolute_timing
        )
    else:
        raise ValueError("You must pass a Musa object or a file to initialize this class.")
    return musa_obj
