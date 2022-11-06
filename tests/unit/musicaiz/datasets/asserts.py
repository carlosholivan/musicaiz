from pathlib import Path
import os
import tempfile
from musicaiz.tokenizers import MMMTokenizer


def _assert_tokenize(dataset_path, dataset, args):
    # create temp ouput file that will be deleted after the testing
    with tempfile.TemporaryDirectory() as output_path:
        # tokenize
        output_file = "token-sequences"
        dataset.tokenize(
            dataset_path=dataset_path,
            output_path=output_path,
            output_file=output_file,
            args=args,
            tokenize_split="all"
        )

        # check that train, validation and test paths exist and contain a txt
        assert Path(output_path, "train", output_file + ".txt").is_file()
        assert Path(output_path, "validation", output_file + ".txt").is_file() 
        assert Path(output_path, "test", output_file + ".txt").is_file()

        # check that txt in validation path is not empty
        assert os.path.getsize(Path(output_path, "train", output_file + ".txt")) > 0

        # get vocabulary and save it in `dataset_path`
        vocab = MMMTokenizer.get_vocabulary(
            dataset_path=output_path
        )
        assert len(vocab) != 0