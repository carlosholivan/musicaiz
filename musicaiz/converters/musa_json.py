from typing import Union, List
from pathlib import Path
import json
from dataclasses import dataclass


from musicaiz.loaders import Musa


@dataclass
class NoteJSON:
    start: int  # ticks
    end: int  # ticks
    pitch: int
    velocity: int


@dataclass
class BarJSON:
    time_sig: str
    start: int  # ticks
    end: int  # ticks
    notes: List[NoteJSON]


@dataclass
class InstrumentJSON:
    is_drum: bool
    name: str
    n_prog: int
    bars: List[BarJSON]


    
@dataclass
class JSON:
    tonality: str
    time_sig: str
    instruments: List[InstrumentJSON]


class MusaJSON:

    """
    This class converst a `musicaiz` :func:`~musicaiz.loaders.Musa` object
    into a JSON format.
    
    Examples
    --------

    >>> file = Path("../0.mid")
    >>> midi = Musa(file, structure="bars", absolute_timing=True)
    >>> musa_json = MusaJSON(midi)

    To add a field inside an instrument:

    >>> musa_json.add_instrument_field(
            n_program=0,
            field="hello",
            value=2
        )
    
    Save the json to disk:

    >>> musa_json.save("filename")
    """

    def __init__(
        self,
        musa_obj: Musa,
    ):
        self.midi = musa_obj
        self.json = self.to_json(musa_obj=self.midi)
    
    def save(self, filename: str, path: Union[str, Path] = ""):
        """Saves the JSON into disk."""
        with open(Path(path, filename + ".json"), "w") as write_file:
            json.dump(self.json, write_file)
    
    @staticmethod
    def to_json(musa_obj):
        composition = {}

        # headers
        composition["tonality"] = musa_obj.tonality
        composition["time_sig"] = musa_obj.time_sig.time_sig
        composition["instruments"] = [{}] * len(musa_obj.instruments)

        composition["instruments"] = []
        for i, instr in enumerate(musa_obj.instruments):
            composition["instruments"].append(
                {
                    "is_drum": instr.is_drum,
                    "name": instr.name,
                    "n_prog": int(instr.program),
                }
            )
            if instr.bars is None:
                continue
            if len(instr.bars) == 0:
                continue
            composition["instruments"][i]["bars"] = []
            for b, bar in enumerate(instr.bars):
                composition["instruments"][i]["bars"].append(
                    {
                        "time_sig": bar.time_sig,
                        "start": bar.start_ticks,
                        "end": bar.end_ticks
                    }
                )
                composition["instruments"][i]["bars"][b]["notes"] = []
                if len(bar.notes) == 0:
                    continue
                for n, note in enumerate(bar.notes):
                    composition["instruments"][i]["bars"][b]["notes"].append(
                        {
                            "start": note.start_ticks,
                            "end": note.end_ticks,
                            "pitch": note.pitch,
                            "velocity": note.velocity,
                        }
                    )
        return composition

    def add_instrument_field(self, n_program: int, field: str, value: Union[str, int, float]):
        """
        Adds a new key - value pair to the instrument which n_program is equal to the
        input ``n_program``.

        Parameters
        ----------

        n_program: int

        field: str

        value: Union[str, int, float]
        """
        self.__check_n_progr(n_program)
        for instr in self.json["instruments"]:
            if n_program != instr["n_prog"]:
                continue
            instr.update({str(field): value})
        self.json

    def delete_instrument_field():
        pass

    def __check_n_progr(self, n_program: int):
        """
        Checks if the input ``n_program`` is in the current json.

        Parameters
        ----------

        n_program: int
            The program number corresponding to the instrument.

        Raises
        ------

        ValueError: _description_
        """
        progrs = [instr["n_prog"] for instr in self.json["instruments"]]
        # check if n_prog exists in the current json
        if n_program not in progrs:
            raise ValueError(f"The input n_program {n_program} is not in the current json. The n_programs of the instruments in the current json are {progrs}.")

    def add_bar_field():
        pass

    def delete_bar_field():
        pass

    def add_note_field():
        pass

    def delete_note_field():
        pass

    def add_header_field():
        pass

    def delete_header_field():
        pass


class JSONMusa:
    pass