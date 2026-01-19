import argparse
import os
from time import time
import torch
import torch as th
from accelerate import Accelerator
from diffusers.models import AutoencoderKL
from accelerate.utils import set_seed
from dataset import P2IDataset
from model import DiT_models
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def create_network(config):
    return DiT_models[config.model_type](
        img_resolution=config.image_size // config.f,
        in_channels=config.num_in_channels,
        out_channels=config.num_out_channels
    )

def train(args):
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    batch_size = args.batch_size

    dataset=P2IDataset(data_dir=args.data_dir,batch_size=args.batch_size,is_inference=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    model = create_network(args).to(device, dtype=dtype)

    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    first_stage_model = first_stage_model.eval()
    first_stage_model.train = False
    for param in first_stage_model.parameters():
        param.requires_grad = False

    dino_v2_vitb14 = th.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device, dtype=dtype) 
    dino_v2_vitb14.eval()
    dino_v2_vitg14 = th.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device, dtype=dtype)
    dino_v2_vitg14.eval()
    for param in dino_v2_vitb14.parameters():
        param.requires_grad = False
    for param in dino_v2_vitg14.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    data_loader, model, optimizer, scheduler = accelerator.prepare(data_loader, model, optimizer, scheduler)

    global_step, epoch, init_epoch = 0, 0, 0
    start_time = time()
    log_steps = 0
    loss_list=[]

    for epoch in tqdm(range(init_epoch, args.num_epoch)):
        tot_loss=0.0
        tot_iter=0
        for iteration, batch in enumerate(data_loader):
            for key, value in batch.items():
                if key!='path' and key!='target_path':
                    batch[key]=batch[key].to(device, dtype=dtype, non_blocking=True)
            for train_flip in range(0,2):
                model.zero_grad()
                if train_flip==0:
                    cond={}
                    target = batch['target_image']
                    source = batch['source_image']
                    target_pose=batch['target_pose_image']
                    cond['dino_tgt_pose'] = batch['dino_pose_tgt']
                    cond['dino_src_pose'] = batch['dino_pose_src']
                    cond['dino_src_img'] = batch['dino_src']
                    cond['dino_tgt_img'] = batch['dino_tgt']
                else:
                    cond = {}
                    target = batch['source_image']
                    source = batch['target_image']
                    target_pose=batch['source_pose_image']
                    cond['dino_tgt_pose'] = batch['dino_pose_src']
                    cond['dino_src_pose'] = batch['dino_pose_tgt']
                    cond['dino_src_img'] = batch['dino_tgt']
                    cond['dino_tgt_img'] = batch['dino_src']

                z_0 = first_stage_model.encode(target).latent_dist.sample().mul_(args.scale_factor)
                cond['src_img']=first_stage_model.encode(source).latent_dist.sample().mul_(args.scale_factor)
                cond['target_pose']=first_stage_model.encode(target_pose).latent_dist.sample().mul_(args.scale_factor)

                tmp = dino_v2_vitb14.get_intermediate_layers(cond['dino_tgt_pose'], 1,
                                                             return_class_token=True)[0]
                cond['dino_tgt_pose'] = th.cat((tmp[1].unsqueeze(1), tmp[0]), dim=1)  
                tmp = dino_v2_vitg14.get_intermediate_layers(cond['dino_src_img'], 1, return_class_token=True)[0]
                cond['dino_src_img'] = th.cat((tmp[1].unsqueeze(1), tmp[0]), dim=1)

                tmp = 0

                t = torch.rand((z_0.size(0),), dtype=dtype, device=device)
                t = t.view(-1, 1, 1, 1)
                z_1 = torch.randn_like(z_0)

                z_t = (1 - t) * z_0 + (1e-5 + (1 - 1e-5) * t) * z_1
                u = (1 - 1e-5) * z_1 - z_0


                input = torch.cat([z_t, cond['src_img'], cond['target_pose']], dim=1)
                # estimate velocity
                v = model(t.squeeze(), input,dino_tgt_pose=cond['dino_tgt_pose'], src_img=cond['src_img'],dino_src_img=cond['dino_src_img'],target_pose=cond['target_pose'])
                loss = F.mse_loss(v, u)
                tot_loss += loss
                tot_iter += 1
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1
                log_steps += 1
                if iteration % 100 == 0 and train_flip==1:
                    if accelerator.is_main_process:
                        end_time = time()
                        steps_per_sec = log_steps / (end_time - start_time)
                        accelerator.print(
                            "epoch {} iteration{}, Loss: {}, Train Steps/Sec: {:.2f}".format(
                                epoch, iteration, loss.item(), steps_per_sec
                            )
                        )
                        log_steps = 0
                        start_time = time()
        loss_list.append(tot_loss/tot_iter)
        if not args.no_lr_decay:
            scheduler.step()
        if epoch%100==99:
            torch.save(model.state_dict(), f'model_params_epoch{epoch}.pth')
    torch.save(model.state_dict(), 'model_params.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')
    torch.save(scheduler.state_dict(), 'scheduler.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument(
        "--model_type",
        type=str,
        default="DiT-L/2",
        help="model_type",
        choices=[
            "DiT-B/2",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-XL/2",
        ],
    )
    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=12, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of model")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,8,4),
        help="resolution of applying attention",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--data_dir", default="/home/ubuntu/dataset/deepfashion")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate g")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)
    args = parser.parse_args()
    train(args)
