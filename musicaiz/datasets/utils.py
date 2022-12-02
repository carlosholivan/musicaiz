from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
from typing import List, Optional, Dict, Type
from rich.console import Console


from musicaiz.tokenizers import (
    MMMTokenizer,
    MMMTokenizerArguments,
    REMITokenizer,
    REMITokenizerArguments,
    TOKENIZER_ARGUMENTS,
    TokenizerArguments,
)


def tokenize_path(
    dataset_path: str,
    dest_path: str,
    metadata: Optional[Dict],
    output_file: str,
    args: Type[TokenizerArguments],
) -> None:

    text_file = open(Path(dest_path, output_file + ".txt"), "w")

    n_jobs = mp.cpu_count()
    pool = mp.Pool(n_jobs)
    results = []

    if metadata is not None:
        elements = metadata.keys()
        total = len(list(metadata.keys()))
    else:
        elements = dataset_path.rglob("*.mid")
        elements = [f.name for f in dataset_path.rglob("*.mid")]
        total = len(elements)

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
                        args
                    )
                ),
                "file": el
            }
        )

    pbar = tqdm(
        results,
        total=total,
        bar_format="{l_bar}{bar:10}{r_bar}",
        colour="GREEN"
    )
    for result in pbar:
        console = Console()
        console.print(f'Processing file [bold orchid1]{result["file"]}[/bold orchid1]')
        res = result["result"].get()
        if res is not None:
            text_file.write(res)

    pool.close()
    pool.join()


def _processer(
    metadata: Optional[Dict],
    data_piece: Path,
    dataset_path: Path,
    args: Type[TokenizerArguments],
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
        prev_tokens = ""
        # Tokenize
        file = Path(dataset_path, data_piece)
        if metadata is not None:
            if "composer" in metadata[data_piece].keys():
                prev_tokens = f"COMPOSER={metadata[data_piece]['composer']} "
            if "period" in metadata[data_piece].keys():
                prev_tokens += f"PERIOD={metadata[data_piece]['period']} "
            if "genre" in metadata[data_piece].keys():
                prev_tokens += f"GENRE={metadata[data_piece]['genre']}"

        if type(args) not in TOKENIZER_ARGUMENTS:
            raise ValueError("Non valid tokenizer args object.")
        if isinstance(args, MMMTokenizerArguments):
            args.prev_tokens = prev_tokens
            tokenizer = MMMTokenizer(file, args)
            piece_tokens = tokenizer.tokenize_file()
        elif isinstance(args, REMITokenizerArguments):
            args.prev_tokens = prev_tokens
            tokenizer = REMITokenizer(file, args)
            piece_tokens = tokenizer.tokenize_file()
        else:
            raise ValueError("Non valid tokenizer.")
        piece_tokens += "\n"

        return piece_tokens
    except:
        pass
