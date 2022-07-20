import numpy as np
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import trange
import json
from typing import Union, Optional
import argparse

from musicaiz.models.transformer_composers.dataset import get_vocabulary
from musicaiz.models.transformer_composers.train import initialize_model
from musicaiz.models.transformer_composers.configs import GPTConfigs
from musicaiz.tokenizers.mmm import MMMTokenizer


def indices_to_text(indices, vocabulary):
    return " ".join([vocabulary[index] for index in indices])


def top_k_logits(logits, k):
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


def sample_sequence(
    dataset_path: Union[Path, str],
    checkpoint_path: Union[Path, str] = Path("results"),
    model_name: str = "gpt",
    dataset_name: str = "maestro",
    start_token: Optional[int] = None, #vocabulary.index("PIECE_START")
    batch_size: int = 1,
    context: str = "PIECE_START",
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = 'cpu',
    sample: bool = True,
    seq_len: int = 512,
    save_midi: bool = False,
    file_path: str = ""
) -> str:

    """
    This function generates a sequence from a pretrained model. The condition to generate the sequence is
    to store the pretrained model as `model_name.pth` in the directory `modelname_datasetname`.
    The dataset path must be provided to look at the `vocabulary.txt` file that allows to convert the token
    indexes to the token names.
    Another thing to consider is that, when saving the midi file the `time_unit` must be the same of the ones that
    the training dataset so the midi will be generated with the correct timing information.

    Parameters
    ----------

    dataset_path: Union[Path, str]

    checkpoint_path: Union[Path, str] = Path("results")

    model_name: str = "gpt"

    dataset_name: str = "maestro"

    start_token: Optional[int] = None

    batch_size: int = 1

    context: str = "PIECE_START"

    temperature: float = 1.0

    top_k: int = 0

    device: str = 'cpu'

    sample: bool = True

    seq_len: int = 512

    save_midi: bool = False

    file_path: str = ""

    Returns
    -------

    token_seq: str
        The generated token sequence.
    """
    # TODO: STOP generation if a PAD is generated
    vocabulary = get_vocabulary(dataset_path)
    context = [vocabulary.index(context)]
    # Get configs file
    model_path = Path(checkpoint_path, model_name + "_" + dataset_name)
    configs_file = Path(model_path, model_name + "_configs.json")
    with open(configs_file) as file:
        configs = json.load(file)

    model = initialize_model(model_name, configs, device)
    model.to(device)
    model.load_state_dict(torch.load(Path(model_path, model_name + ".pth"))["model_state_dict"])
    model.eval()

    if start_token is None:
        assert context is not None, 'You must give the start_token or the context'
        context = torch.tensor(context, device=device, dtype=torch.float16).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'You must give the start_token or the context'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.float16)

    prev = context
    output = context
    with torch.no_grad():
        for i in trange(seq_len): #trange(configs["model_configs"]["SEQ_LEN"]):
            logits = model(output)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    output = output.to(torch.int)

    logging.info(f"Generated token seq indexes: {output.tolist()[0]}")

    token_seq = indices_to_text(output.tolist()[0], vocabulary)
    logging.info(f"Generated token seq is: {token_seq}")

    if save_midi:
        data = list(token_seq.split(" "))
        # TODO: be careful with the time unit in which TIME_DELTA tokens are represented
        midi_obj = MMMTokenizer.tokens_to_musa(data, absolute_timing=True, time_unit="HUNDRED_TWENTY_EIGHT")
        midi_obj.write_midi(file_path)
        logging.info(f"Saved midi file to: {token_seq}")

    return token_seq

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        help="",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="",
        required=True,
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        help="",
        required=False,
        default=GPTConfigs.SEQ_LEN,
    )
    parser.add_argument(
        "--save_midi",
        type=bool,
        help="",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--file_path",
        type=str,
        help="",
        required=False,
        default="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sample_sequence(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        seq_len=args.sequence_length,
        save_midi=args.save_midi,
        file_path=args.file_path,
    )
