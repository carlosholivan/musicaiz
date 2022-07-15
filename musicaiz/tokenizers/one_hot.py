"""

    OneHot.one_hot

"""
import numpy as np
from typing import List
import matplotlib.pyplot as plt


# Our modules
from musicaiz.structure import Note


class OneHot:

    # TODO: Initialize with time axis ticks or secs?
    """
    This class ....

    """

    @staticmethod
    def one_hot(
        notes: List[Note],
        min_pitch: int = 0,
        max_pitch: int = 127,
        time_axis: str = "ticks",
        step: int = 10,
        vel_one_hot: bool = True,
    ) -> np.array:

        """
        Each pitch from `min_pitch` to `max_pitch` has a value of 0
        (if the pitch value is not being played in the current time step) or 1
        (if the pitch value is being played in the current time step)

        Parameters
        ----------

        Raises
        ------
        ValueError
            if ...

        Examples
        --------
        Decompose a magnitude spectrogram into 32 components with NMF

        >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
        >>> S = np.abs(librosa.stft(y))
        >>> comps, acts = librosa.decompose.decompose(S, n_components=8)
        Sort components by ascending peak frequency

        """

        if time_axis == "ticks":
            max_time = int(notes[-1].end_ticks / step)
            loop_step = step
        elif time_axis == "seconds":
            max_time = int(notes[-1].end_sec / step)
            loop_step = int(step * 1000)  # step in for loop must be in ms
        else:
            raise ValueError("Not a valid axis. Axist must be 'seconds' or 'ticks'.")

        if step > max_time:
            raise ValueError(f"Step value {step} must be smaller than the total time steps {max_time}.")

        # Initialize one hot array
        pitch_range = max_pitch - min_pitch
        one_hot = np.zeros((pitch_range, max_time))
        for time_step in range(0, max_time, loop_step):
            for note in notes:
                if time_axis == "ticks":
                    note_start = note.start_ticks
                    note_end = note.end_ticks
                elif time_axis == "seconds":
                    note_start = note.start_sec * 1000
                    note_end = note.end_sec * 1000

                if (note_start <= time_step <= note_end) and (min_pitch <= note.pitch <= max_pitch):
                    if vel_one_hot:
                        one_hot[note.pitch - min_pitch, time_step] = note.velocity
                    else:
                        one_hot[note.pitch - min_pitch, time_step] = 1
        return one_hot

    def notes_to_one_hot():
        """Same as before but each instrument in the 3rd axis"""
        pass

    # TODO: Move this to plotters module?
    @staticmethod
    def plot(one_hot_tensor: np.array):
        plt.subplots(figsize=(20, 5))
        aspect = int(one_hot_tensor.shape[1] / one_hot_tensor.shape[0] / 10)
        if aspect <= 0:
            aspect = 1
        plt.imshow(one_hot_tensor, origin="lower", aspect=aspect)
        plt.xlabel("Time")
        plt.ylabel("Pitch")
