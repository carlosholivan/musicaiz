import pytest


from musicaiz.tokenizers import (
    CPWordTokenizer,
    CPWordTokenizerArguments
)


@pytest.fixture
def cpword_tokens(fixture_dir):
    tokens_path = fixture_dir / "tokenizers" / "cpword_tokens.txt"
    text_file = open(tokens_path, "r")
    # read whole file to a string
    yield text_file.read()


@pytest.fixture
def midi_sample(fixture_dir):
    return fixture_dir / "tokenizers" / "mmm_tokens.mid"


def test_CPWordTokenizer_tokenize(midi_sample, cpword_tokens):
    args = CPWordTokenizerArguments(sub_beat="SIXTEENTH")
    tokenizer = CPWordTokenizer(midi_sample, args)
    tokens = tokenizer.tokenize_file()
    assert tokens == cpword_tokens
