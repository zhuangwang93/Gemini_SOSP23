import json
import sys
import os
import time
from argparse import ArgumentParser

import deepspeed
from deepspeed.utils.logging import logger
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (DataCollatorForLanguageModeling, BertConfig,
                          BertForMaskedLM, BertTokenizerFast)

sys.path.append('../')
from utils import get_argument_parser, is_time_to_exit, set_snapshot_settings


global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0


def count_parameters(model):
    return sum(p.ds_numel for p in model.parameters())


def print_at_rank0(msg):
    if dist.get_rank() == 0:
        print(msg)


def checkpoint_model(PATH, ckpt_id, model, epoch, last_global_step,
                     last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        'epoch': epoch,
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.save_checkpoint(PATH, ckpt_id,
                                            checkpoint_state_dict)
    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)
    logger.info(status_msg)
    return



def load_training_checkpoint(args, model, PATH, ckpt_id):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    _, checkpoint_state_dict = model.load_checkpoint(PATH, ckpt_id)
    epoch = checkpoint_state_dict['epoch']
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict[
        'last_global_data_samples']
    del checkpoint_state_dict
    return (epoch, last_global_step, last_global_data_samples)



def load_checkpoint(args, model):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}, CKPT_ID={args.load_checkpoint_id}"
    )
    start_epoch, global_step, global_data_samples = load_training_checkpoint(
        args=args,
        model=model,
        PATH=args.load_training_checkpoint,
        ckpt_id=args.load_checkpoint_id)
    logger.info(
        f"The model is loaded from last checkpoint at epoch {start_epoch} when the global steps were at {global_step} and global data samples at {global_data_samples}"
    )

    if args.rewarmup:
        logger.info(
            f"Rewarmup learning rate with last_global_step_from_restore = {global_step}"
        )
        last_global_step_from_restore = global_step

    return start_epoch


def get_args():
    parser = get_argument_parser()
    deepspeed.add_config_arguments(parser)
    args = parser.parse_args(sys.argv[1:])

    config = json.load(open(args.deepspeed_config, 'r', encoding='utf-8'))
    args.job_name = config['name'] if args.job_name is None else args.job_name

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/", args.job_name)
    args.config = config

    return args


class FakeDataset(Dataset):
    """temporarily using this to avoid having to load a real dataset"""

    def __init__(self, epoch_sz: int) -> None:
        self.__epoch_sz = epoch_sz

    def __getitem__(self, _: int) -> dict:
        return {"text": "Hello, my dog is cute, this is a fake sentence"}

    def __len__(self) -> int:
        return self.__epoch_sz


class CollatorForLMWrapper(DataCollatorForLanguageModeling):
    def __init__(self, device, max_length: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.max_length = max_length

    def __call__(self, examples):
        batch = self.tokenizer(
            [e["text"] for e in examples],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
            return_tensors="pt"
        )

        batch = list(
            map(dict, zip(*[[(k, v) for v in batch[k]] for k in batch.keys()]))
        )

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.cuda()
        return batch



def get_dataloader(args, tokenizer): 
    train_dataset = FakeDataset(200000)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    micro_bs = args.config['train_micro_batch_size_per_gpu']

    training_loader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        collate_fn=CollatorForLMWrapper(
                tokenizer=tokenizer,
                device=torch.device(f"cuda:{args.local_rank}"),
                max_length=512,
                mlm=False,
        ),
        sampler=DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            seed=1234,
        ),
        )

    return training_loader



def get_model(args) -> Module:
    model_config = args.config['bert_config']

    cfg = BertConfig(
        max_position_embeddings=model_config['max_position_embeddings'],
        type_vocab_size=model_config['type_vocab_size'],
        num_attention_heads=model_config['num_attention_heads'],
        num_hidden_layers=model_config['num_hidden_layers'],
        hidden_size=model_config['hidden_size'],
        intermediate_size=model_config['intermediate_size'],
        # gradient_checkpointing=model_config['gradient_checkpointing'],
        vocab_size=model_config['vocab_size'],
    )

    if args.config['zero_optimization']['stage'] == 3:
        with deepspeed.zero.Init(config=args.config):
            model = BertForMaskedLM(cfg)
    else:
        model = BertForMaskedLM(cfg)

    model.gradient_checkpointing_enable()
    time.sleep(2)
    print_at_rank0(model)
    print_at_rank0(f"model param size {count_parameters(model)/1e9} B")
    time.sleep(5)
    return model



def main():
    args = get_args()
    model = get_model(args)
    print_at_rank0(f"[total number of parameters] {count_parameters(model)}")

    cpu_snapshot, snapshot_settings, training_profiler, snapshot_profiler, comm_profiler = set_snapshot_settings(args)
    
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print_at_rank0(f'tokenizer vocab_size {tokenizer.vocab_size}'
            f' config vocab size {args.config["bert_config"]["vocab_size"]}')
    assert tokenizer.vocab_size <= args.config["bert_config"]["vocab_size"]
    # get data_loader
    training_dataloader = get_dataloader(args, tokenizer)

    model, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )

    global global_step
    global global_data_samples

    if not None in [args.load_training_checkpoint, args.load_checkpoint_id]:
        print(f"load checkpoint from {args.load_training_checkpoint}")
        training_profiler.record_checkpoint_load_start_time()
        start_epoch = load_checkpoint(args, model)
        training_profiler.record_checkpoint_load_end_time()
    elif args.pre_checkpoint:
        checkpoint_model(PATH=args.saved_model_path,
                        ckpt_id=args.snapshot_path,
                        model=model,
                        epoch=1,
                        last_global_step=global_step,
                        last_global_data_samples=global_data_samples)

    for e in range(10):
        snapshot_settings.set_global_epoch(e + 1)
        for n, inputs in enumerate(training_dataloader):
            if n < 5 and e < 1:
                print_at_rank0(f"{[inputs[k].size() for k in inputs]}")
            step_start = time.time()

            outputs = model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs["input_ids"])
            loss = outputs.loss
            model.backward(loss) 
            model.step()
            
            if model.is_gradient_accumulation_boundary():
                print_at_rank0(f"{e} {n}, LOSS: {loss.item()}")

            loss = None

            step_time = time.time() - step_start
            print_at_rank0(f"[Training] step time: {step_time}")

            global_step += 1
            snapshot_settings.set_global_step(global_step)
            if dist.get_rank() == 0:
                if snapshot_settings.is_profile_mode():
                    snapshot_profiler.set_iteration_time(step_time)
                    snapshot_profiler.report()
                    snapshot_profiler.reset()
                comm_profiler.profile_comm_time_gap()

            if is_time_to_exit(args=args, global_steps=global_step):
                print_at_rank0(
                    f'Warning: Early training termination due to max steps limit, epoch={e+1}, global_step={global_step}'
                )
                return

            

if __name__ == "__main__":
    main()