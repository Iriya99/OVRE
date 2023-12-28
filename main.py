import os
import time
import torch
import argparse
import builtins
import torch.nn as nn
import os.path as osp
import torch.nn.functional as nnf
import torch.distributed as dist
from utils import *
from optim import *
from tqdm import tqdm
from attention import *
from einops import repeat
from transformers import CLIPModel, GPT2LMHeadModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--video_sample_type", default="uniform", help="'rand'/'uniform'")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=44)
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--prefix_length", type=int, default=64)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--clip_emb_size", type=int, default=768)
    parser.add_argument("--gpt_emb_size", type=int, default=768)
    parser.add_argument("--clip_lr", type=float, default=1e-6)
    parser.add_argument("--gpt_lr", type=float, default=2e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--freeze_visual", type=str2bool, default=False)
    parser.add_argument("--sep_tok", type=str, default="<|triplet|>")
    parser.add_argument("--version", type=str, default="cls")
    parser.add_argument("--load_epoch", type=int, default=50)
    args = parser.parse_args()
    return args

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_tokens("<|triplet|>")
    return tokenizer

class MMIT(Dataset):

    def __getitem__(self, index):
        vix = self.data[index]["vix"]
        video = load_frames(osp.join(f"{self.args.root}", "data", "images", f"{vix}"), "uniform", self.args.num_frames)
        video = self.transform(video)
        gts = ""
        for x in self.data[index]["gts"]:
            gts += x + self.args.sep_tok
        gts = gts[:-len(self.args.sep_tok)] + "<|endoftext|>"
        gts_tokens = torch.tensor(self.tokenizer.encode(gts))
        padding = self.args.max_seq_len - gts_tokens.shape[0]
        if padding > 0:
            gts_tokens = torch.cat((gts_tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            gts_tokens = gts_tokens[:self.args.max_seq_len]
        mask = gts_tokens.ge(0)
        gts_tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.args.prefix_length), mask), dim=0)
        return vix, video, gts_tokens, mask
    
    def __init__(self, args, mode="train"):
        self.args = args
        self.tokenizer = get_tokenizer()
        self.transform = get_augmentation(mode, args)
        if mode == "train":
            self.data = load_pkl(osp.join(f"{self.args.root}", "data", "train.pkl"))
        else:
            self.data = load_pkl(osp.join(f"{self.args.root}", "data", "test.pkl"))
        
    def __len__(self):
        return len(self.data)

def load_mmit(args):
    train_set = MMIT(args, "train")
    train_sampler = DistributedSampler(train_set, shuffle=True) if torch.distributed.is_initialized() else None
    train_loader = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True, num_workers=16)

    test_set = MMIT(args, "test")  
    test_sampler = DistributedSampler(test_set, shuffle=False) if torch.distributed.is_initialized() else None
    test_loader = DataLoader(dataset=test_set, batch_size=args.bs, shuffle=False, sampler=test_sampler, drop_last=True, num_workers=16)

    return train_loader, test_loader

class Ovre(nn.Module):

    def forward(self, video, gts_tokens, mask=None):
        video = video.reshape(-1, 3, self.args.input_size, self.args.input_size)
        video_queries = repeat(self.video_queries, "n d -> b n d", b=self.args.bs)
        patch_embeds = self.clip.get_image_features(video)[0].reshape(self.args.bs, -1, self.args.clip_emb_size) # modify the code in modeling_clip.py to make the CLIP model return patch features
        prefix_embeds = self.attn_pooler(video_queries, patch_embeds)
        embedding_gts = self.gpt.transformer.wte(gts_tokens)
        embedding_cat = torch.cat((prefix_embeds, embedding_gts), dim=1) 
        out = self.gpt(inputs_embeds=embedding_cat, attention_mask=mask)
        cap_logits = out.logits[:, self.args.prefix_length-1: -1]
        loss = nnf.cross_entropy(cap_logits.reshape(-1, cap_logits.shape[-1]), gts_tokens.flatten(), ignore_index=0)
        return loss, prefix_embeds

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = get_tokenizer()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False if args.freeze_visual else True
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt.resize_token_embeddings(len(self.tokenizer))
        self.video_queries = nn.Parameter(torch.randn(args.prefix_length, args.clip_emb_size))
        self.attn_pooler = CrossAttention(dim=args.gpt_emb_size, context_dim=args.clip_emb_size)

def load_model(args):
    model_path = osp.join(f"{args.root}", "checkpoints", f"{args.version}_{args.load_epoch}.pt")
    model = Ovre(args)
    if osp.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:1")), strict=False)
    else:
        print(f"{model_path} is not exist")
    return model

def main():
    args = get_args()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    
    if local_rank != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    model = Ovre(args).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    params_optimizer = list(model.named_parameters())
    gpt_params = [p for n, p in params_optimizer if "gpt." in n]
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    other_params = [p for n, p in params_optimizer if "clip." not in n and "gpt." not in n]
    optimizer_grouped_params = [{"params": clip_params, "lr": args.clip_lr}, {"params": gpt_params, "lr": args.gpt_lr}, {"params": other_params, "lr": args.lr}]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    train_loader, test_loader = load_mmit(args)
    num_training_steps = len(train_loader) * args.bs
    num_warmup_steps = int(args.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    epochs = args.epochs
    if args.mode == "train":
        for epoch in range(1, 1+epochs):
            train(train_loader, model, epoch, optimizer, scheduler, args)
            if epoch % args.save_every == 0:
                torch.save(model.module.state_dict(), osp.join(args.out_dir, f"{args.version}_{epoch}.pt"))
    else:
        model = load_model(args).cuda()
        test(test_loader, model, args)

def train(train_loader, model, epoch, optimizer, scheduler, args):
    
    model.train()
    print("==> training...\n")
    batch_time = AverageMeter("Time", ":1.2f")
    loss_log = AverageMeter("loss", ":2.2f")
    progress = ProgressMeter(
    len(train_loader),
    [batch_time, loss_log],
    prefix="Epoch: [{}]".format(epoch))
    end = time.time()  
    for i, (vix, video, gts_tokens, mask) in enumerate(train_loader):   
        model.zero_grad()
        video, gts_tokens, mask = video.cuda(), gts_tokens.cuda(), mask.cuda()
        loss, _ = model(video, gts_tokens, mask)
        loss_log.update(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
def test(test_loader, model, args):
    wb = ""
    for i, (vix, video, gts_tokens, mask) in enumerate(tqdm(test_loader)):
        video, gts_tokens, mask = video.cuda(), gts_tokens.cuda(), mask.cuda()
        _, prefix_embed = model(video,   gts_tokens, mask)
        generated = generate_beam(model, model.tokenizer, embed=prefix_embed)[0]
        gens = generated[11:].replace("<|endoftext|>", "").split(args.sep_tok)
        triplets = ""
        for gen in gens:
            triplets = triplets + gen + " / "
        triplets = triplets[:-3]
        # print(f"{int(vix)} gen:{triplets}")
        wb = wb + f"{int(vix)}||{triplets}\n"

    with open(osp.join(f"{args.root}", "results", f"{args.version}_{args.load_epoch}.txt"), "w+") as f:
        f.write(wb)

def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=44,
    temperature=1.0,
    stop_token: str = "<|endoftext|>",
):

    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

main()