from typing import List, Dict, Union, Optional
import numpy as np
from dataclasses import dataclass

from musicaiz.rhythm import TimingConsts, ms_per_tick


@dataclass
class QuantizerConfig:
    """
    Basic quantizer arguments.

    Parameters
    ----------

    note: str
        The note length of the grid.

    strength: parameter between 0 and 1.
        Example GRID = [0 24 48], STAR_TICKS = [3 ,21, 40] and Aq
        START_NEW_TICS = [(3-0)*strength, (21-24)*strength, (40-48)*strength]
        END_NEW_TICKS = []

    delta_qr: Q_range in ticks

    type_q: type of quantization
        if negative: only differences between start_tick and grid > Q_r is
        taking into account for the quantization. If positive only differences
        between start_tick and grid < Q_r is taking into accounto for the quantization.
        If none all start_tick is quantized based on the strength (it works similar to basic
        quantization but adding the strength parameter)
    """
    delta_qr: int = 12
    strength: int = 1  # 100%
    type_q: Optional[str] = None


def _find_nearest(
    array: List[Union[int, float]], value: Union[int, float]
) -> Union[int, float]:
    """Find de array component value closest to a given value
    [3, 6, 9, 12] 5 --> 6"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_ticks_from_subdivision(
    subdivisions: List[Dict[str, Union[int, float]]]
) -> List[int]:
    """Extract the grid array in ticks from a subdivision"""
    v_grid = []
    for i in range(len(subdivisions)):
        v_grid.append(subdivisions[i].get("ticks"))

    return v_grid


def basic_quantizer(
    input_notes,
    grid: List[int],
    bpm: int = TimingConsts.DEFAULT_BPM.value
):
    for i in range(len(input_notes)):
        start_tick = input_notes[i].start_ticks
        end_tick = input_notes[i].end_ticks
        start_tick_quantized = _find_nearest(grid, start_tick)

        delta_tick = start_tick - start_tick_quantized

        if delta_tick > 0:
            input_notes[i].start_ticks = start_tick - delta_tick
            input_notes[i].end_ticks = end_tick - delta_tick

        elif delta_tick < 0:
            input_notes[i].start_ticks = start_tick + abs(delta_tick)
            input_notes[i].end_ticks = end_tick + abs(delta_tick)

        input_notes[i].start_sec = input_notes[i].start_ticks * ms_per_tick(bpm)
        input_notes[i].end_sec = input_notes[i].end_ticks * ms_per_tick(bpm)


def advanced_quantizer(
    input_notes,
    grid: List[int],
    config: QuantizerConfig,
    bpm: int = TimingConsts.DEFAULT_BPM.value,
    resolution: int = TimingConsts.RESOLUTION.value,
):
    """
    This function quantizes a musa object given a grid.

    Parameters
    ----------

    file: musa object

    grid: array of ints in ticks
    """

    Aq = config.strength

    for i in range(len(input_notes)):

        start_tick = input_notes[i].start_ticks
        end_tick = input_notes[i].end_ticks
        start_tick_quantized = _find_nearest(grid, start_tick)
        delta_tick = start_tick - start_tick_quantized
        delta_tick_q = int(delta_tick * Aq)

        if config.type_q == "negative" and (abs(delta_tick) > config.delta_qr):
            if delta_tick > 0:
                input_notes[i].start_ticks = start_tick - delta_tick_q
                input_notes[i].end_ticks = end_tick - delta_tick_q
            else:
                if delta_tick < 0:
                    input_notes[i].start_ticks = start_tick + abs(delta_tick_q)
                    input_notes[i].end_ticks = end_tick + abs(delta_tick_q)

        elif config.type_q == "positive" and (abs(delta_tick) < config.delta_qr):
            if delta_tick > 0:
                input_notes[i].start_ticks = input_notes[i].start_ticks - delta_tick_q
                input_notes[i].end_ticks = input_notes[i].end_ticks - delta_tick_q
            else:
                if delta_tick < 0:
                    input_notes[i].start_ticks = input_notes[i].start_ticks + abs(delta_tick_q)
                    input_notes[i].end_ticks = input_notes[i].end_ticks + abs(delta_tick_q)

        elif config.type_q is None:
            if delta_tick > 0:
                input_notes[i].start_ticks = input_notes[i].start_ticks - delta_tick_q
                input_notes[i].end_ticks = input_notes[i].end_ticks - delta_tick_q
            else:
                if delta_tick < 0:
                    input_notes[i].start_ticks = input_notes[i].start_ticks + abs(delta_tick_q)
                    input_notes[i].end_ticks = input_notes[i].end_ticks + abs(delta_tick_q)

        input_notes[i].start_sec = input_notes[i].start_ticks * ms_per_tick(bpm, resolution) / 1000
        input_notes[i].end_sec = input_notes[i].end_ticks * ms_per_tick(bpm, resolution) / 1000
