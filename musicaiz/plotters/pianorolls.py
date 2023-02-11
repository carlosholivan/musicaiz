from re import sub
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from typing import List, Union, Optional
from pathlib import Path
import warnings


# Our modules
from musicaiz.rhythm import (
    get_subdivisions,
    TimingConsts
)
from musicaiz.structure import Note, Instrument
from musicaiz.loaders import Musa

# TODO: Add more colors. This only handles 10 instruments per plot
COLOR_EDGES = [
    '#C232FF',
    '#89FFAE',
    '#FFFF8B',
    '#A9E3FF',
    '#FF9797',
    '#A5FFE8',
    '#FDB0F8',
    '#FFDC9C',
    '#F3A3C4',
    '#E7E7E7',
    "#AA7DBB",
    "#7DBB90",
    "#83C0B9",
    "#83AFC0",
    "#8AA7C3",
    "#8A94C3",
    "#C793CB",
    "#CB93B9",
    "#CA97A0",
    "#CAB697",
]

COLOR = [
    '#D676FF',
    '#0AFE57',
    '#FEFF00',
    '#56C8FF',
    '#FF4C4C',
    '#4CFFD1',
    '#FF4CF4',
    '#FFB225',
    '#C25581',
    '#737D73',
    "#E3A4FB",
    "#A7F6BF",
    "#A7F6ED",
    "#ADE4F9",
    "#ADD4F9",
    "#B5C1F9",
    "#F5B5F9",
    "#FEBDE9",
    "#FEBDC9",
    "#F3DFC0",
]


# TODO: subdivisions in plotly
# TODO: Hide note by clicking pitch in legend plotly
# TODO: method to save plot in png or html


class Pianoroll:

    """

    Arguments
    ---------

    dark: bool = True
        If we want the plot with a dark background.

    Examples
    --------

    >>> from pathlib import Path
    >>> file = Path("tests/fixtures/tokenizers/mmm_tokens.mid")
    >>> musa_obj = musicaiz.loaders.Musa(file, structure="instruments")
    >>> plot = musicaiz.plotters.Pianoroll()
    >>> plot.plot_instrument(
            track=musa_obj.instruments[0].notes,
            total_bars=3,
            subdivision="eight"
        )
    """

    def __init__(
        self,
        musa: Optional[Musa] = None,
        dark: bool = False,
    ):

        if dark:
            background_color = "#282828"
        else:
            background_color = "#EEEEEE"

        fig, self.ax = plt.subplots(figsize=(20, 5), dpi=300)
        self.ax.yaxis.set_major_locator(MultipleLocator(12))

        self.ax.set_facecolor(background_color)
        plt.xlabel("Time (bar)")

        self.musa = musa

    def plot_grid(self, subdivisions):
        # TODO: If we have lots of subdivisions (subdivision arg is small), then
        # the x ticks cannot be seen properly, maybe it's better in that case to represent
        # only the tick for each beat.
        # TODO: The same happens with pitch (y ticks). We should make smth to avoid
        # all the pitch values to be written in the axis
        plt.xlim((0, len(subdivisions) - 1))
        # Each subdivision has a vertical lines (grid)
        prev_bar_idx = 0
        labels = []
        for s in subdivisions:
            if s.bar_idx != prev_bar_idx:
                labels.append(str(s.bar_idx))
            else:
                labels.append("")
            prev_bar_idx = s.bar_idx
        self.ax.set_xticks([s.start_ticks for s in subdivisions])
        self.ax.set_xticklabels(labels)
        self.ax.xaxis.grid(which="major", linewidth=0.1, color="gray")
        # Get labels for bar and beats
        prev_bar, prev_beat = 0, 0
        bars_labels, beats_labels = [], []
        for s in subdivisions:
            if s.bar_idx != prev_bar and not prev_bar == 0:
                bars_labels.append(s)
                self.ax.axvline(x=s.start_ticks, linestyle="-", linewidth=0.6, color="grey")
            if s.beat_idx != prev_beat and not prev_beat == 0:
                beats_labels.append(s)
                self.ax.axvline(x=s.start_ticks, linestyle="-", linewidth=0.2, color="grey")
            prev_bar, prev_beat = s.bar_idx, s.beat_idx

    def _notes_loop(self, notes: List[Note], idx: int):
        plt.ylabel("Pitch")
        #highest_pitch = get_highest_pitch(track.instrument)
        #lowest_pitch = get_lowest_pitch(track.instrument)
        #  plt.ylim((0, (highest_pitch - lowest_pitch) - 1))

        for note in notes:
            plt.vlines(x=note.start_ticks,
                       ymin=note.pitch,
                       ymax=note.pitch + 1,
                       color=COLOR_EDGES[idx],
                       linewidth=0.01)

            self.ax.add_patch(
                plt.Rectangle((note.start_ticks, note.pitch),
                              width=note.end_ticks - note.start_ticks,
                              height=1,
                              alpha=note.velocity / 127,
                              edgecolor=COLOR_EDGES[idx],
                              facecolor=COLOR[idx]))

    def plot_instruments(
        self,
        program: Union[int, List[int]],
        bar_start: int,
        bar_end: int,
        print_measure_data: bool = True,
        show_bar_labels: bool = True,
        show_grid: bool = True,
        show: bool = False,
        save: bool = False,
        save_path: Optional[Union[Path, str]] = None,
    ):

        if print_measure_data:
            plt.text(
                x=0, y=1.3, s=f"Measure: {self.musa.time_signature_changes[0]}", transform=self.ax.transAxes,
                horizontalalignment='left', verticalalignment='top', fontsize=12)

            plt.text(
                0, 1.2, f"Displayed bars: {self.musa.total_bars}", transform=self.ax.transAxes,
                horizontalalignment='left', verticalalignment='top', fontsize=12)

            plt.text(
                0, 1.1, f"Quantized: {self.musa.is_quantized}", transform=self.ax.transAxes,
                horizontalalignment='left', verticalalignment='top', fontsize=12)

            plt.text(
                1, 1.3, f"Tempo: {self.musa.tempo_changes[0]}bpm", transform=self.ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12)

            plt.text(
                1, 1.2, f"Subdivision: {self.musa.subdivision_note}", transform=self.ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12)

        subdivisions = self.musa.get_subbeats_in_bars(bar_start, bar_end)
        if show_grid:
            self.plot_grid(subdivisions)
        notes = self.musa.get_notes_in_bars(bar_start, bar_end, program)
        if isinstance(program, list):
            for i, p in enumerate(program):
                notes_i = self.musa._filter_by_instruments(p, None, notes)
                if notes_i is not None:
                    self._notes_loop(notes_i, i)
        else:
            self._notes_loop(notes, 0)
        if not show_bar_labels:
            self.ax.get_yaxis().set_visible(False)
            self.ax.get_xaxis().set_visible(False)
        if show:
            plt.show()
        if save:
            plt.margins(0,0)
            plt.subplots_adjust(
                top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0
            )
            plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight', pad_inches=0)


class PianorollHTML:

    def __init__(
        self,
        musa: Optional[Musa] = None,
    ):

        """The Musa object need to be initialized with the argument
        structure = `bars` for a good visualization of the pianoroll."""

        self.musa = musa
        #ax.yaxis.set_major_locator(MultipleLocator(1))
        #ax.grid(linewidth=0.25)
        #ax.set_facecolor('#282828')
        self.fig = go.Figure(
            data=go.Scatter(),
            layout=go.Layout(
                {
                    "title": "",
                    #"template": "plotly_dark",
                    "xaxis": {'title': "subdivisions (bar)"}, 
                    "yaxis": {'title': 'pitch'},
                }
            )
        )

    def _notes_loop(self, notes: List[Note], idx: int):
        for note in notes:
            self.fig.add_shape(
                type="rect",
                x0=note.start_ticks,
                y0=note.pitch,
                x1=note.end_ticks,
                y1=note.pitch + 1,
                line=dict(
                    color=COLOR_EDGES[idx],
                    width=2,
                ),
                fillcolor=COLOR[idx],
            )

            # this is to add a hover information on each note
            self.fig.add_trace(
                go.Scatter(
                    x=[
                        note.start_ticks,
                        note.start_ticks,
                        note.end_ticks,
                        note.end_ticks,
                        note.start_ticks
                    ],
                    y=[
                        note.pitch,
                        note.pitch + 1,
                        note.pitch + 1,
                        note.pitch,
                        note.pitch
                    ],
                    fill="toself",
                    mode="lines",
                    name=f"pitch={note.pitch}<br>\n"
                         f"velocity={note.velocity}<br>\n"
                         f"start_ticks={note.start_ticks}<br>\n"
                         f"end_ticks={note.end_ticks}<br>",
                    opacity=0,
                    showlegend=False,
                )
            )

    def plot_grid(self, subdivisions):
        # all the pitch values to be written in the axis
        #self.fig.update_xaxes(range[0, len(subdivisions) - 1])
        #plt.xlim((0, len(subdivisions) - 1))
        # Each subdivision has a vertical lines (grid)
        #self.fig.set_xticks([s["ticks"] for s in subdivisions])
        ##labels = [str(s["ticks"]) for s in subdivisions]
        prev_bar_idx = 0
        labels = []
        for s in subdivisions:
            if s.bar_idx != prev_bar_idx:
                labels.append(str(s.bar_idx))
            else:
                labels.append("")
            prev_bar_idx = s.bar_idx
        self.fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[s.start_ticks for s in subdivisions],
                ticktext=labels,
                tickfont=dict(
                    size=10,
                ),
            ),
        )
        #self.fig.xaxis.grid(which="major", linewidth=0.1, color="gray")
        # Get labels for bar and beats
        prev_bar, prev_beat = 0, 0
        bars_labels, beats_labels = [], []
        for s in subdivisions:
            if s.bar_idx != prev_bar and not prev_bar == 0:
                bars_labels.append(s)
                self.fig.add_vline(x=s.start_ticks, line_width=0.6, line_color="grey")
            if s.beat_idx != prev_beat and not prev_beat == 0:
                beats_labels.append(s)
                self.fig.add_vline(x=s.start_ticks, line_width=0.2, line_color="grey")
            prev_bar, prev_beat = s.bar_idx, s.beat_idx

    def plot_instruments(
        self,
        program: Union[int, List[int]],
        bar_start: int,
        bar_end: int,
        path: Union[Path, str] = Path("."),
        filename: str = "title",
        save_plot: bool = True,
        show_grid: bool = True,
        show: bool = True
    ):

        subdivisions = self.musa.get_subbeats_in_bars(bar_start, bar_end)
        if show_grid:
            self.plot_grid(subdivisions)
        notes = self.musa.get_notes_in_bars(bar_start, bar_end, program)
        if isinstance(program, list):
            for i, p in enumerate(program):
                notes_i = self.musa._filter_by_instruments(p, None, notes)
                if notes_i is not None:
                    self._notes_loop(notes_i, i)
        else:
            self._notes_loop(notes, 0)

        pitches = [note.pitch for note in notes]

        # this is to add the yaxis labels# horizontal line for pitch grid
        labels = [i for i in range(min(pitches) - 1, max(pitches) + 2)]
        for pitch in labels:
            self.fig.add_hline(y=pitch + 1, line_width=0.1, line_color="white")
        # if we do have too many pitches, we won't label all of them in the yaxis
        if max(pitches) - min(pitches) > 24:
            cleaned_labels = [label for label in labels if label % 2 == 0]
        else:
            cleaned_labels = labels

        # Adjust y labels (pitch)
        # TODO: label between pitches
        self.fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=cleaned_labels,
                ticktext=cleaned_labels,
                tickfont=dict(
                    size=12,
                ),
            ),
        )

        self.fig.update_layout(legend={
            "xanchor": "center", "yanchor": "top"
        })

        if save_plot:
            self.fig.write_html(Path(path, filename + ".html"))

        if show:
            self.fig.show()
