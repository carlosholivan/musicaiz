import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from typing import List, Union
from pathlib import Path
import warnings


# Our modules
from musicaiz.rhythm import (
    get_subdivisions,
    TimingConsts
)
from musicaiz.structure import Note, Instrument


COLOR_EDGES = [
    '#C232FF',
    '#C232FF',
    '#89FFAE',
    '#FFFF8B',
    '#A9E3FF',
    '#FF9797',
    '#A5FFE8',
    '#FDB0F8',
    '#FFDC9C',
    '#F3A3C4',
    '#E7E7E7'
]

COLOR = [
    '#D676FF',
    '#D676FF',
    '#0AFE57',
    '#FEFF00',
    '#56C8FF',
    '#FF4C4C',
    '#4CFFD1',
    '#FF4CF4',
    '#FFB225',
    '#C25581',
    '#737D73'
]


# TODO: subdivisions in plotly
# TODO: Plot all instruments in one plot plt and plotly
# TODO: Hide note by clicking pitch in legend plotly
# TODO: plot bars plotly
# TODO: rethink bar plotting in plt
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

    def __init__(self, dark: bool = False):

        if dark:
            background_color = "#282828"
        else:
            background_color = "#EEEEEE"

        fig, self.ax = plt.subplots(figsize=(20, 5), dpi=300)
        self.ax.yaxis.set_major_locator(MultipleLocator(12))

        self.ax.set_facecolor(background_color)
        plt.xlabel("Time (bar.beat.subdivision)")

    def plot_grid(self, subdivisions):
        # TODO: If we have lots of subdivisions (subdivision arg is small), then
        # the x ticks cannot be seen properly, maybe it's better in that case to represent
        # only the tick for each beat.
        # TODO: The same happens with pitch (y ticks). We should make smth to avoid
        # all the pitch values to be written in the axis
        plt.xlim((0, len(subdivisions) - 1))
        # Add 1st subdivision of new bar after last bar for better plotting the last bar
        self._add_new_bar_subdiv(subdivisions)
        # Each subdivision has a vertical lines (grid)
        self.ax.set_xticks([s["ticks"] for s in subdivisions])
        ##labels = [str(s["ticks"]) for s in subdivisions]
        labels = [str(s["bar"]) + "." + str(s["bar_beat"]) + "." + str(s["bar_subdivision"]) for s in subdivisions]
        self.ax.set_xticklabels(labels)
        self.ax.xaxis.grid(which="major", linewidth=0.1, color="gray")
        # Get labels for bar and beats
        prev_bar, prev_beat = 0, 0
        bars_labels, beats_labels = [], []
        for s in subdivisions:
            if s["bar"] != prev_bar and not prev_bar == 0:
                bars_labels.append(s)
                self.ax.axvline(x=s["ticks"], linestyle="--", linewidth=0.4, color="red")
            if s["bar_beat"] != prev_beat and not prev_beat == 0:
                beats_labels.append(s)
                self.ax.axvline(x=s["ticks"], linestyle="--", linewidth=0.2, color="blue")
            prev_bar, prev_beat = s["bar"], s["bar_beat"]

    @staticmethod
    def _add_new_bar_subdiv(subdivisions):
        """In `rhythm.get_subdivisions` method, we get all the subdivisions (starting times).
        We can add the 1st subdivision of a new bar after the subdivisions dict for plotting."""
        range_sec = subdivisions[-1]["sec"] - subdivisions[-2]["sec"]
        range_ticks = subdivisions[-1]["ticks"] - subdivisions[-2]["ticks"]
        subdivisions.append({
            "bar": subdivisions[-1]["bar"] + 1,
            "piece_beat": 1,
            "piece_subdivision": 1,
            "bar_beat": 1,
            "bar_subdivision": 1,
            "sec": subdivisions[-1]["sec"] + range_sec,
            "ticks": subdivisions[-1]["ticks"] + range_ticks,
        })

    def _notes_loop(self, notes: List[Note]):
        plt.ylabel("Pitch")
        #highest_pitch = get_highest_pitch(track.instrument)
        #lowest_pitch = get_lowest_pitch(track.instrument)
        #  plt.ylim((0, (highest_pitch - lowest_pitch) - 1))

        for note in notes:
            plt.vlines(x=note.start_ticks,
                       ymin=note.pitch,
                       ymax=note.pitch + 1,
                       color=COLOR_EDGES[0],
                       linewidth=0.01)

            self.ax.add_patch(
                plt.Rectangle((note.start_ticks, note.pitch),
                              width=note.end_ticks - note.start_ticks,
                              height=1,
                              alpha=note.velocity / 127,
                              edgecolor=COLOR_EDGES[0],
                              facecolor=COLOR[0]))

    def plot_instrument(
        self,
        track,
        total_bars: int,
        subdivision: str,
        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
        quantized: bool = False,
        print_measure_data: bool = True,
        show_bar_labels: bool = True
    ):

        if print_measure_data:
            plt.text(
                x=0, y=1.3, s=f"Measure: {time_sig}", transform=self.ax.transAxes,
                horizontalalignment='left', verticalalignment='top', fontsize=12)

            plt.text(
                0, 1.2, f"Displayed bars: {total_bars}", transform=self.ax.transAxes,
                horizontalalignment='left', verticalalignment='top', fontsize=12)

            plt.text(
                0, 1.1, f"Quantized: {quantized}", transform=self.ax.transAxes,
                horizontalalignment='left', verticalalignment='top', fontsize=12)

            plt.text(
                1, 1.3, f"Tempo: {bpm}bpm", transform=self.ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12)

            plt.text(
                1, 1.2, f"Subdivision: {subdivision}", transform=self.ax.transAxes,
                horizontalalignment='right', verticalalignment='top', fontsize=12)

            #plt.text(
                #1, 1.1, f"Instrument: {track.name}", transform=self.ax.transAxes,
                #horizontalalignment='right', verticalalignment='top', fontsize=12)

        subdivisions = get_subdivisions(total_bars, subdivision, time_sig, bpm, resolution)
        self.plot_grid(subdivisions)
        self._notes_loop(track)
        if not show_bar_labels:
            self.ax.get_yaxis().set_visible(False)
            self.ax.get_xaxis().set_visible(False)


class PianorollHTML:

    """The Musa object need to be initialized with the argument
    structure = `bars` for a good visualization of the pianoroll."""

    #ax.yaxis.set_major_locator(MultipleLocator(1))
    #ax.grid(linewidth=0.25)
    #ax.set_facecolor('#282828')
    fig = go.Figure(
        data=go.Scatter(),
        layout=go.Layout(
            {
                "title": "",
                #"template": "plotly_dark",
                "xaxis": {'title': "subdivisions (bar.beat.subdivision)"}, 
                "yaxis": {'title': 'pitch'},
            }
        )
    )

    def _notes_loop(self, notes: List[Note]):
        for note in notes:
            self.fig.add_shape(
                type="rect",
                x0=note.start_ticks,
                y0=note.pitch,
                x1=note.end_ticks,
                y1=note.pitch + 1,
                line=dict(
                    color=COLOR_EDGES[0],
                    width=2,
                ),
                fillcolor=COLOR[0],
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
        # Add 1st subdivision of new bar after last bar for better plotting the last bar
        Pianoroll._add_new_bar_subdiv(subdivisions)
        # Each subdivision has a vertical lines (grid)
        #self.fig.set_xticks([s["ticks"] for s in subdivisions])
        ##labels = [str(s["ticks"]) for s in subdivisions]
        labels = [str(s["bar"]) + "." + str(s["bar_beat"]) + "." + str(s["bar_subdivision"]) for s in subdivisions]
        self.fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[s["ticks"] for s in subdivisions],
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
            if s["bar"] != prev_bar and not prev_bar == 0:
                bars_labels.append(s)
                self.fig.add_vline(x=s["ticks"], line_width=0.4, line_color="red")
            if s["bar_beat"] != prev_beat and not prev_beat == 0:
                beats_labels.append(s)
                self.fig.add_vline(x=s["ticks"], line_width=0.2, line_color="blue")
            prev_bar, prev_beat = s["bar"], s["bar_beat"]

    def plot_instrument(
        self,
        track: Instrument,
        bar_start: int,
        bar_end: int,
        subdivision: str,
        path: Union[Path, str] = Path("."),
        filename: str = "title",
        save_plot: bool = True,
        time_sig: str = TimingConsts.DEFAULT_TIME_SIGNATURE.value,
        bpm: int = TimingConsts.DEFAULT_BPM.value,
        resolution: int = TimingConsts.RESOLUTION.value,
        show: bool = True
    ):

        pitches = []
        if track.bars is None:
            warnings.warn("Track has no bars. You probably initialized the Musa object with structure=`instruments`. \n \
                          You can use `structure=`bars` to get the track bars. \n \
                          The plotter is going to ignore the bars and it'll plot all the notes in the track.")
            self._notes_loop(track.notes)
            for note in track.notes:
                if note.pitch not in pitches:
                    pitches.append(note.pitch)
        else:
            # TODO
            for bar in track.bars[bar_start:bar_end]:
                self._notes_loop(bar.notes)
                for note in bar.notes:
                    if note.pitch not in pitches:
                        pitches.append(note.pitch)

            total_bars = bar_end - bar_start
            subdivisions = get_subdivisions(total_bars, subdivision, time_sig, bpm, resolution)
            self.plot_grid(subdivisions)

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
        # TODO: label in the middle of the pitch
        self.fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=labels,
                ticktext=labels,
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
