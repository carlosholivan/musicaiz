from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Optional, Dict

from musicaiz.tokenizers import MMMTokenizer


def tokenize_path(
    dataset_path: str,
    dest_path: str,
    metadata: Optional[Dict],
    output_file: str
):

    text_file = open(Path(dest_path, output_file + ".txt"), "w")

    n_jobs = mp.cpu_count()
    pool = mp.Pool(n_jobs)
    results = []

    if metadata is not None:
        elements = metadata.keys()
        total = len(list(metadata.keys()))
    else:
        elements = dataset_path.rglob("*.mid")
        total = len(list(dataset_path.glob("*.mid")))

    for el in elements:
        # Some files in LMD hace errors (OSError: data byte must be in range 0..127),
        # so we avoid parsing those files
        results.append(
            {
                "result": pool.apply_async(
                    _processer,
                    args=(
                        metadata,
                        el,
                        dataset_path,
                        "MMM"
                    )
                ),
                "file": el
            }
        )

    pbar = tqdm(
        results,
        total=total,
        bar_format="{l_bar}{bar:10}{r_bar}"
    )
    for result in pbar:
        pbar.set_postfix_str(f'Processing file {result["file"]}')
        res = result["result"].get()
        if res is not None:
            text_file.write(result["result"].get())

    pool.close()
    pool.join()


# TODO: args here to be able pass this fn all the args of the MMMTokenizer.tokenize_file
def _processer(
    metadata: Optional[Dict],
    data_piece: Path,
    dataset_path: Path,
    tokenizer: str = "MMM"
) -> List[str]:

    """

    Parameters
    ----------

    data_piece: pathlib.Path
        The path to the midi file.
    
    data_piece: pathlib.Path
        The parent path where the midi file is.


    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    
    try:
        # Tokenize
        file = Path(dataset_path, data_piece)
        if tokenizer == "MMM":
            piece_tokens = MMMTokenizer.tokenize_file(
                file=file,
                prev_tokens="",
                windowing=True,
                time_unit="HUNDRED_TWENTY_EIGHT",
                num_programs=None,
                shuffle_tracks=True,
                track_density=False,
                window_size=4,
                hop_length=1,
                time_sig=True,
                velocity=True
            )
        else:
            raise ValueError("Non valid tokenizer.")
        piece_tokens += "\n"

        return piece_tokens
    except:
        pass