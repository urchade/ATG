import argparse
import logging
import os
import pickle

import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from generate import Evaluator
from model import IeGenerator
from preprocess import GraphIEData
from save_load import save_model

# logging level
logging.basicConfig(level=logging.INFO)


def evaluate(model, eval_loader):
    model.eval()
    evaluator = Evaluator(model=model, loader=eval_loader)
    return evaluator.evaluate()


def train(model, optimizer, train_data, eval_data,
          train_batch_size=32, eval_batch_size=32,
          n_epochs=None, n_steps=None, warmup_ratio=0.1,
          grad_accumulation_steps=1,
          max_num_samples=1,
          save_interval=1000, log_dir="logs"):
    model.train()

    # initialize data loaders
    num_samples = max_num_samples
    trd = GraphIEData(train_data, type='train', max_num_samples=num_samples)
    evd = GraphIEData(eval_data, type='eval')
    train_loader = model.create_dataloader(trd, batch_size=train_batch_size, shuffle=True)
    eval_loader = model.create_dataloader(evd, batch_size=eval_batch_size, shuffle=False)

    device = next(model.parameters()).device

    n_steps = max(len(train_loader) * n_epochs, n_steps)
    n_epochs = n_steps // len(train_loader)

    logging.info(f"Number of epochs: {n_epochs}")
    logging.info(f"Number of steps: {n_steps}")
    logging.info(f"Number of samples: {num_samples}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * warmup_ratio),
        num_training_steps=n_steps
    )

    train_loader_iter = iter(train_loader)

    pbar = tqdm(range(n_steps))
    best_path = None
    best_f1 = 0
    for step in pbar:
        try:
            batch = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device)
        try:
            loss = model(batch)
        except:
            continue

        loss = loss / grad_accumulation_steps

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

        torch.nn.utils.clip_grad_value_(model.token_rep.parameters(), 0.1)

        if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == n_steps:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        description = f'Step {step + 1}/{n_steps}, Epoch {step // len(train_loader) + 1}/{n_epochs}, Loss {loss.item():.4f}, Num Samples {num_samples}'

        pbar.set_description(description)

        if (step + 1) % save_interval == 0:
            # Create log directory
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Evaluate
            metric_dict, metrics = evaluate(model, eval_loader)

            # Save metrics
            with open(os.path.join(log_dir, 'log_metrics.txt'), 'a') as f:
                f.write(f'{description}\n\n')
                f.write(f'{metrics}\n\n\n')

            # current f1 for Strict + not Symetric evaluation
            current_f1 = float(metric_dict["Strict + not Symetric"]["f_score"])

            if current_f1 > best_f1:

                # save current best model
                current_path = os.path.join(log_dir, f'model_{step + 1}_{current_f1:.4f}.pt')
                save_model(model, current_path)

                if best_path is not None:
                    os.remove(best_path)
                best_path = current_path
                best_f1 = current_f1

            model.train()


MODELS = {
    "spanbert": f"/gpfswork/rech/pds/upa43yu/models/spanbert-base-cased",
    "bert": f"/gpfswork/rech/pds/upa43yu/models/bert-base-cased",
    "roberta": f"/gpfswork/rech/pds/upa43yu/models/roberta-base",
    "scibert": f"/gpfswork/rech/pds/upa43yu/models/scibert-base",
    "arabert": f"/gpfswork/rech/pds/upa43yu/models/bert-base-arabert",
    "bertlarge": f"/gpfsdswork/dataset/HuggingFace_Models/bert-large-cased",
    "scibert_cased": f"/gpfswork/rech/pds/upa43yu/models/scibert_cased",
    "albert": f"/gpfswork/rech/pds/upa43yu/models/albert-xxlarge-v2",
    "spanbertlarge": f"/gpfswork/rech/pds/upa43yu/models/spanbert-large-cased",
    "t5-s": "/gpfsdswork/dataset/HuggingFace_Models/t5-small",
    "t5-m": "/gpfsdswork/dataset/HuggingFace_Models/t5-base",
    "t5-l": "/gpfsdswork/dataset/HuggingFace_Models/t5-large",
    "deberta": "/gpfswork/rech/pds/upa43yu/models/deberta-v3-large"
}


# #training_arguments
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='dataset/scierc.pkl')
    parser.add_argument('--model_name', type=str, default='scibert_cased')
    parser.add_argument('--max_width', type=int, default=14)
    parser.add_argument('--num_prompts', type=int, default=5)
    parser.add_argument('--hidden_transformer', type=int, default=512)
    parser.add_argument('--num_transformer_layers', type=int, default=6)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--span_mode', type=str, default='conv_share')
    parser.add_argument('--p_drop', type=float, default=0.1)
    parser.add_argument('--use_pos_code', type=bool, default=True)
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--lr_encoder', type=float, default=1e-5)
    parser.add_argument('--lr_decoder', type=float, default=1e-4)
    parser.add_argument('--lr_others', type=float, default=5e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--max_num_samples', type=int, default=1)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--cross_attn', type=bool, default=True)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Running on the GPU")
    else:
        import flair

        flair.device = torch.device('cpu')
        device = torch.device("cpu")
        print("Running on the CPU")

    # Open the file
    with open(args.data_file, 'rb') as f:
        datasets = pickle.load(f)

    # Load mappings
    class_to_id = datasets['span_to_id']  # entity to id mapping
    rel_to_id = datasets['rel_to_id']  # relation to id mapping
    rel_to_id["stop_entity"] = len(rel_to_id)  # add a new relation for stop entity

    model = IeGenerator(
        class_to_id, rel_to_id, model_name=MODELS[args.model_name], max_width=args.max_width,
        num_prompts=args.num_prompts, hidden_transformer=args.hidden_transformer,
        num_transformer_layers=args.num_transformer_layers, attention_heads=args.attention_heads,
        span_mode=args.span_mode, use_pos_code=args.use_pos_code, p_drop=args.p_drop, cross_attn=args.cross_attn
    )

    model.to(device)

    optimizer = torch.optim.Adam([
        # encoder
        {'params': model.token_rep.parameters(), 'lr': args.lr_encoder},

        # decoder
        {'params': model.decoder.parameters(), 'lr': args.lr_decoder},

        # lstm
        {'params': model.rnn.parameters(), 'lr': args.lr_encoder},

        # projection layers
        {'params': model.project_memory.parameters(), 'lr': args.lr_others},
        {'params': model.project_queries.parameters(), 'lr': args.lr_others},
        {'params': model.project_tokens.parameters(), 'lr': args.lr_others},
        {'params': model.span_rep.parameters(), 'lr': args.lr_others},
        {'params': model.project_span_class.parameters(), 'lr': args.lr_others},
        {'params': model.embed_proj.parameters(), 'lr': args.lr_others},
    ])

    train(
        model=model, optimizer=optimizer, train_data=datasets['train'], eval_data=datasets['dev'],
        train_batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        n_epochs=args.n_epochs, n_steps=args.n_steps, warmup_ratio=args.warmup_ratio,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_num_samples=args.max_num_samples,
        save_interval=args.save_interval, log_dir=args.log_dir
    )
