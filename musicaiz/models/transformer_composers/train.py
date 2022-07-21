import logging
from datetime import datetime
import warnings
from tqdm import tqdm
try:
    from apex import amp
except ImportError:
    amp = None

import json
from pathlib import Path
import argparse
from typing import Union
import csv
from prettytable import PrettyTable
from typing import Dict

import torch
from torch.optim import AdamW
from torch import nn

# gpt_composer sub-package
from musicaiz.models.transformer_composers.configs import (
    GPTConfigs,
    TrainConfigs,
)
from musicaiz.models.transformer_composers.dataset import (
    build_torch_loaders,
    get_vocabulary
)
from musicaiz.models.transformer_composers.transformers import (
    GPT2,
)


def initialize_model(
    model_name: str,
    configs: Dict[str, Union[str, int]],
    device: str
):
    confs = configs["model_configs"]
    if model_name == "gpt":
        model = GPT2(
            vocab_size=confs["VOCAB_SIZE"],
            embedding_dim=confs["EMBED_DIM"],
            n_decoders=confs["N_DECODERS"],
            sequence_len=confs["SEQ_LEN"],
            n_heads=confs["N_HEADS"],
            device=device
        )
    return model

def train(
    dataset_path: Union[str, Path],
    sequence_length: int = GPTConfigs.SEQ_LEN,
    batch_size: int = TrainConfigs.BATCH_SIZE,
    train_split: float = TrainConfigs.TRAIN_SPLIT,
    is_splitted: bool = False,
    epochs: int = TrainConfigs.EPOCHS,
    lr: float = TrainConfigs.LR,
    adam_epsilon: float = TrainConfigs.ADAM_EPSILON,
    gradient_accumulation_steps: float = TrainConfigs.GRAD_ACUM_STEPS,
    fp16: bool = TrainConfigs.FP16,
    fp16_opt_level: str = TrainConfigs.FP16_OPT_LEVEL,
    log_steps: int = TrainConfigs.LOG_STEPS,
    ckpt_steps: int = TrainConfigs.CKPT_STEPS,
    checkpoint_path: Union[str, Path] = TrainConfigs.CHECKPOINT_PATH,
    log_dir: Union[str, Path] = TrainConfigs.LOG_DIR,
    model_name: str = TrainConfigs.MODEL_NAME
):

    """
    
    Parameters
    ----------

    dataset_path: Union[str, Path]

    sequence_length: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2Configs.SEQ_LEN`

    batch_size: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2Configs.BATCH_SIZE`

    embedding_dim: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2Configs.EMBED_DIM`

    n_decoders: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2Configs.N_DECODERS`

    n_heads: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2Configs.N_HEADS`

    dropout: float
        Default is :func:`~musicaiz.models.gpt_composer.GPT2Configs.DROPOUT`

    train_split: float
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.TRAIN_SPLIT`

    epochs: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.EPOCHS`

    lr: float
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.LR`

    adam_epsilon: float
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.ADAM_EPSILON`

    gradient_accumulation_steps: float
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.GRAD_ACUM_STEPS`

    fp16: bool
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.FP16`

    fp16_opt_level: str
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.FP16_OPT_LEVEL`

    log_steps: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.LOG_STEPS`

    ckpt_steps: int
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.CKPT_STEPS`

    checkpoint_path: Union[str, Path]
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.CHECKPOINT_PATH`

    log_dir: Union[str, Path]
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.LOG_DIR`

    model_name: str
        Default is :func:`~musicaiz.models.gpt_composer.GPT2TrainConfigs.MODEL_NAME`
    """

    if isinstance(log_dir, str):
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    logging.getLogger("Train")

    logging.info("Creating dataloaders...")

    train_dataloader, val_dataloader = build_torch_loaders(
        dataset_path=dataset_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        train_split=train_split,
        is_splitted=is_splitted
    )
    logging.info(f"Train samples {len(train_dataloader.dataset)} | Validation samples {len(val_dataloader.dataset)}")
    logging.info(
        f"Train tokens {len(train_dataloader.dataset) * sequence_length} \
            | Validation samples {len(val_dataloader.dataset) * sequence_length}"
    )

    # sample
    sample_idx = next(iter(train_dataloader))
    logging.info(f"Sample len {len(sample_idx)} data {sample_idx}")
    logging.info(f"input shape: {sample_idx.shape}")

    tokens = get_vocabulary(dataset_path)
    vocab_size = len(tokens)
    logging.info(f"Vocabulary size is {vocab_size}")

    # TODO: This needs to go into a separated method
    # Save hyperparams to json
    if model_name == "gpt":
        model_configs = {
            key: value for key, value in GPTConfigs.__dict__.items() if "__" not in key
        }
    model_configs.update({"VOCAB_SIZE": vocab_size})
    train_configs = {
        key: value for key, value in TrainConfigs.__dict__.items() if "__" not in key and not isinstance(value, Path)
    }
    configs_dict = {
        "model_configs": model_configs,
        "train_configs": train_configs,
    }
    dir_configs = Path(str(log_dir), model_name + "_configs.json")
    with open(dir_configs, 'w') as outfile:
        json.dump(configs_dict, outfile)
    logging.info(f"Saved configs.json in {dir_configs}")

    # Initialize model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = initialize_model(model_name, configs_dict, device)
    model.to(device)

    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        return table, total_params
    table, total_params = count_parameters(model)
    print(table)
    logging.info(f"Total trainable params: {total_params}")

    # write model params in txt
    dir_params = Path(str(log_dir), model_name + "_model.txt")
    with open(dir_params, 'a+') as results_file:
        results_file.write(f"{table} \n Total trainable params: {total_params}")
        results_file.close()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}
    ]

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=lr,
        eps=adam_epsilon
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_dataloader) // gradient_accumulation_steps,
        epochs=epochs,
        pct_start=0.05,
        anneal_strategy='linear'
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    losses = {}
    global_steps = 0
    local_steps = 0
    step_loss = 0.0
    train_loss = 0.0
    train_perplexity = 0.0
    start_epoch = 0
    start_step = 0
    step_perplexity = 0.0

    model.train()

    logging.info(f'{datetime.now()} | Moved model to: {device}')
    logging.info(f'{datetime.now()} | train_batch_size: {batch_size} | eval_batch_size: {batch_size}')
    logging.info(f'{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps}')
    logging.info(f'{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}')

    model.zero_grad()  # Reset gradients tensors
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):  # tqdm(range(epochs), desc='Epochs', position=0):
        logging.info(f"{datetime.now()} | Epoch: {epoch} \n")
        pb = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            bar_format="{l_bar}{bar:10}{r_bar}"
        )
        for step, batch in pb:
            if step < start_step:
                continue
            inputs = batch  # _ is input_mask
            inputs = inputs.to(device)
            lm_logits = model(inputs)
            shift_logits = lm_logits[..., :-1, :].contiguous()

            # if we don't use softmax at the end of last layer
            loss = loss_fn(shift_logits.view(-1, vocab_size), inputs[:, 1:].contiguous().view(-1).type(torch.LongTensor).to(device))

            step_perplexity = torch.exp(loss)
            origin_loss = loss.item()

            loss = loss / gradient_accumulation_steps  # divide loss into gradient accumulation step
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            step_loss = origin_loss
            losses[global_steps] = origin_loss

            train_loss += loss.item()
            train_perplexity += step_perplexity.item()

            local_steps += 1
            global_steps += 1

            if global_steps % gradient_accumulation_steps == 0:
                if fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_norm=1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scheduler.step()
                optimizer.step()
                model.zero_grad()
            
            total_train_loss = train_loss / local_steps
            total_train_perplexity = train_perplexity / local_steps

            step_loss = 0.0
            local_steps = 0

            dir_train = Path(str(log_dir), model_name + "_train_results.txt")
            with open(dir_train, 'a+') as results_file:
                json.dump(losses, results_file)
            results_file.close()

            # Evaluate every epoch
            model.eval()

            eval_loss = 0.0
            perplexity = 0.0
            eval_steps = 0

            for step, batch in tqdm(
                enumerate(val_dataloader),
                total=len(val_dataloader),
                bar_format='{l_bar}{bar:10}{r_bar}'
            ):

                inputs = batch  # _ is input_mask
                inputs = inputs.to(device)
                labels = inputs
                labels = labels.to(device)
                lm_logits = model(inputs)

                with torch.no_grad():
                    lm_logits = model(inputs)
                
                shift_logits = lm_logits[..., :-1, :].contiguous()

                # if we don't use spftmax at the end of last layer
                loss = loss_fn(shift_logits.view(-1, vocab_size), inputs[:, 1:].contiguous().view(-1).type(torch.LongTensor).to(device))

                tmp_eval_loss = loss
                tmp_perplexity = torch.exp(tmp_eval_loss)

                eval_loss += tmp_eval_loss.item()
                perplexity += tmp_perplexity.item()
                eval_steps += 1

                total_eval_loss = eval_loss / eval_steps
                total_perplexity = perplexity / eval_steps

                dir_eval = Path(str(log_dir), model_name + "_eval_results.txt")
                with open(dir_eval, 'a+') as results_file:
                    results_file.write(
                        f'{datetime.now()} | Step: {step} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity}\n'
                    )
                    results_file.close()

            # We finished 1 global step
            # write csv with results
            csv_eval = Path(str(log_dir), model_name + "_results.csv")
            with open(csv_eval, 'a+') as results_file:
                writer = csv.writer(results_file)
                if global_steps == 1:
                    writer.writerow(["Step", "Train PPL", "Eval PPL", "Train Loss", "Eval Loss"])
                writer.writerow([global_steps, total_train_loss, total_perplexity, total_train_perplexity, total_eval_loss])
                results_file.close()

            pb.set_postfix_str(
                f"{datetime.now()} | Epoch {epoch} | Global Step: {global_steps} | Train Loss: {origin_loss} | Train PPL: {step_perplexity} \n" \
            )
            pb.set_postfix_str(
                f"{datetime.now()} | Epoch {epoch} | Global Step: {global_steps} | Eval Loss: {total_eval_loss} | Perplexity: {total_perplexity} \n"
            )

            model.train()
            start_step = 0

            if global_steps % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'losses': losses,
                    'train_step': global_steps,
                    'amp': amp.state_dict()
                    }, f'{checkpoint_path}/ep{epoch}_step{global_steps}_{model_name}.pth'
                )
                logging.info(f'\n {datetime.now()} | Saved checkpoint to: {checkpoint_path} \n')
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
                'train_step': global_steps,
                'amp': amp.state_dict()
                }, f'{checkpoint_path}/ep{epoch}_{model_name}.pth'
            )
            logging.info(f'\n {datetime.now()} | Saved checkpoint to: {checkpoint_path} \n')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'losses': losses,
        'train_step': global_steps,
        'amp': amp.state_dict()
        }, f'{checkpoint_path}/ep{epoch}_{model_name}.pth'
    )
    logging.info(f'\n {datetime.now()} | Saved checkpoint to: {checkpoint_path} \n')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
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
        "--batch_size",
        type=int,
        help="",
        required=False,
        default=TrainConfigs.BATCH_SIZE,
    )
    parser.add_argument(
        "--train_split",
        type=float,
        help="",
        required=False,
        default=TrainConfigs.TRAIN_SPLIT,
    )
    parser.add_argument(
        "--is_splitted",
        type=bool,
        help="",
        required=False,
        default=TrainConfigs.IS_SPLITTED,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="",
        required=False,
        default=TrainConfigs.EPOCHS,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="",
        required=False,
        default=TrainConfigs.LR,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        help="",
        required=False,
        default=TrainConfigs.ADAM_EPSILON,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=float,
        help="",
        required=False,
        default=TrainConfigs.GRAD_ACUM_STEPS,
    )
    parser.add_argument(
        "--fp16",
        type=bool,
        help="",
        required=False,
        default=TrainConfigs.FP16,
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        help="",
        required=False,
        default=TrainConfigs.FP16_OPT_LEVEL,
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        help="",
        required=False,
        default=TrainConfigs.LOG_STEPS,
    )
    parser.add_argument(
        "--ckpt_steps",
        type=int,
        help="",
        required=False,
        default=TrainConfigs.CKPT_STEPS,
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="",
        required=False,
        default=TrainConfigs.CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="",
        required=False,
        default=TrainConfigs.LOG_DIR,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="",
        required=False,
        default=TrainConfigs.MODEL_NAME,
    )
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset_path,
        train_split=args.train_split,
        is_splitted=args.is_splitted,
        epochs=args.epochs,
        lr=args.lr,
        adam_epsilon=args.adam_epsilon,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        fp16_opt_level=args.fp16_opt_level,
        log_steps=args.log_steps,
        ckpt_steps=args.ckpt_steps,
        checkpoint_path=args.checkpoint_path,
        log_dir=args.log_dir,
        model_name=args.model_name,
    )
    