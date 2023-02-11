from typing import Tuple, Union, Dict, Any
from pathlib import Path
import pandas as pd
import math
import numpy as np

from musicaiz.rhythm import TimeSignature
from musicaiz.harmony import Tonality, AllChords
from musicaiz.structure import NoteClassBase


class BPSFH:

    TIME_SIGS = {
        "1": "4/4",
        "2": "2/4",
        "3": "4/4",
        "4": "6/8",
        "5": "3/4",
        "6": "2/4",
        "7": "4/4",
        "8": "4/4",
        "9": "4/4",
        "10": "2/4",
        "11": "4/4",
        "12": "3/8",
        "13": "4/4",
        "14": "4/4",
        "15": "3/4",
        "16": "2/4",
        "17": "4/4",
        "18": "3/4",
        "19": "2/4",
        "20": "4/4",
        "21": "4/4",
        "22": "3/4",
        "23": "12/8",
        "24": "2/4",
        "25": "3/4",
        "26": "2/4",
        "27": "3/4",
        "28": "6/8",
        "29": "4/4",
        "30": "2/4",
        "31": "3/4",
        "32": "4/4",
    }

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)

    def parse_anns(
        self,
        anns: str = "high"
    ) -> Dict[str, pd.DataFrame]:
        """
        Converts the bar index float annotations in 8th notes.
        """
        table = {}

        for file in list(self.path.glob("*/*.mid")):
            filename = file.name.split(".")[0]
            table[file.name.split(".")[0]] = []
            anns_path = Path(self.path, file.name.split(".")[0])

            gt = pd.read_excel(
                Path(anns_path, "phrases.xlsx"),
                header=None
            )

            # Read file with musanalysis to get tempo
            time_sig = TimeSignature(self.TIME_SIGS[filename])

            # Loop in rows
            prev_sec = ""
            rows_ans = []
            j = 0
            for i, row in gt.iterrows():
                if anns == "high":
                    sec_name = row[2]
                elif anns == "mid":
                    sec_name = row[3]
                elif anns == "low":
                    sec_name = row[4]
                if i == 0:
                    if row[0] < 0:
                        dec, quarters = math.modf(row[0])
                        if quarters == 0.0:
                            summator = 1
                        elif quarters == -1.0:
                            summator = time_sig.num
                    else:
                        summator = 0
                    if anns == "high":
                        rows_ans.append([ann for k, ann in enumerate(row) if k <= 2])
                    elif anns == "mid":
                        rows_ans.append([ann for k, ann in enumerate(row) if (k <= 1 or k == 3)])
                    elif anns == "low":
                        rows_ans.append([ann for k, ann in enumerate(row) if (k <= 1 or k == 4)])
                    rows_ans[-1][0] += summator
                    rows_ans[-1][1] += summator
                    prev_sec = sec_name
                    j += 1
                    end_time = row[1]
                    continue
                if sec_name != prev_sec:
                    rows_ans[j - 1][1] = end_time + summator
                    if anns == "high":
                        rows_ans.append([ann for k, ann in enumerate(row) if k <= 2])
                    elif anns == "mid":
                        rows_ans.append([ann for k, ann in enumerate(row) if (k <= 1 or k == 3)])
                    elif anns == "low":
                        rows_ans.append([ann for k, ann in enumerate(row) if (k <= 1 or k == 4)])
                    rows_ans[-1][0] += summator
                    rows_ans[-1][1] += summator
                    prev_sec = sec_name
                    j += 1
                if i == len(gt) - 1:
                    rows_ans[-1][1] = row[1] + summator
                end_time = row[1]

            new_df = pd.DataFrame(columns=np.arange(3))
            for i, r in enumerate(rows_ans):
                new_df.loc[i] = r

            table[file.name.split(".")[0]] = new_df

        return table

    @classmethod
    def bpsfh_key_to_musicaiz(cls, note: str) -> Tonality:
        alt = None
        if "-" in note:
            alt = "FLAT"
            note = note.split("-")[0]
        elif "+" in note:
            alt = "SHARP"
            note = note.split("+")[0]
        if note.isupper():
            mode = "MAJOR"
        else:
            mode = "MINOR"
            note = note.capitalize()
        if alt is None:
            tonality = Tonality[note + "_" + mode]
        else:
            tonality = Tonality[note + "_" + alt + "_" + mode]
        return tonality

    @classmethod
    def bpsfh_chord_quality_to_musicaiz(cls, quality: str) -> AllChords:
        if quality == "M":
            q = "MAJOR_TRIAD"
        elif quality == "m":
            q = "MINOR_TRIAD"
        elif quality == "M7":
            q = "MAJOR_SEVENTH"
        elif quality == "m7":
            q = "MINOR_SEVENTH"
        elif quality == "D7":
            q = "DOMINANT_SEVENTH"
        elif quality == "a":
            q = "AUGMENTED_TRIAD"
        return AllChords[q]

    @classmethod
    def bpsfh_chord_to_musicaiz(
        cls,
        note: str,
        degree: int,
        quality: str,
    ) -> Tuple[NoteClassBase, AllChords]:
        tonality = cls.bpsfh_key_to_musicaiz(note)
        qt = cls.bpsfh_chord_quality_to_musicaiz(quality)
        notes = tonality.notes
        return notes[degree - 1], qt
