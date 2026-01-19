import argparse
import os
import torch
from torch import nn
from diffusers.models import AutoencoderKL
from model import DiT_models
from torchdiffeq import odeint_adjoint as odeint
from dataset import P2IDataset
from torch.cuda.amp import autocast
import torchvision
from PIL import Image
from accelerate.utils import set_seed
from accelerate import Accelerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ADAPTIVE_SOLVER = ["dopri5", "dopri8", "adaptive_heun", "bosh3"]
FIXER_SOLVER = ["euler", "rk4", "midpoint", "stochastic"]

accelerator = Accelerator()

class NFECount(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("nfe", torch.tensor(0.0))

    def __call__(self, t, x, *args, **kwargs):
        self.nfe += 1.0
        return accelerator.unwrap_model(self.model).forward_with_cfg(t, x, *args, **kwargs)




def create_network(config):
    return DiT_models[config.model_type](
        img_resolution=config.image_size // config.f,
        in_channels=config.num_in_channels,
        out_channels=config.num_out_channels
    )




def sample_and_test(args):

    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    to_range_0_1 = lambda x: (x + 1.0) / 2.0

    model = create_network(args).to(device)
    dino_v2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
    dino_v2_vitb14.eval()
    dino_v2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to(device)
    dino_v2_vitg14.eval()
    first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device)
    first_stage_model.eval()

    ckpt = torch.load(
        'model_params.pth',
        map_location=device,
    )

    print("Finish loading model")
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    del ckpt


    save_dir = args.save_dir


    dataset = P2IDataset(data_dir=args.data_dir, batch_size=args.batch_size, is_inference=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    if args.nfe_count:
        average_nfe = 0.0
    else:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        average_time=0.0

    if args.method in ADAPTIVE_SOLVER:
        options = {
            "dtype": torch.float64
        }
    else:
        options = {"step_size": args.step_size, "perturb": args.perturb}

    full_iter=0

    if args.nfe_count:
        model = NFECount(model).to(device)  

    data_loader, model = accelerator.prepare(data_loader, model)

    for iteration, batch in enumerate(data_loader):
        print(f'Iteration {iteration:03d}')
        img_s = batch['source_image'].to(device)
        target_pose=batch['target_pose_image'].to(device)
        dino_tgt_pose = batch['dino_pose_tgt'].to(device)
        dino_src_img = batch['dino_src'].to(device)
        target_path=batch['target_path']
        path = batch['path']

        for i in range(len(target_path)):
            gt=Image.open(target_path[i])
            gt=gt.resize((256,256))
            gt.save(f'/home/ubuntu/result/gt/{path[i]}')


        if not args.nfe_count:
            starter.record()

        with torch.no_grad():
            z_target_pose = first_stage_model.encode(target_pose).latent_dist.sample().mul_(args.scale_factor)
            src_img = first_stage_model.encode(img_s, return_dict=True)[0].sample() * 0.18215
            with autocast(dtype=torch.bfloat16):
                tmp = dino_v2_vitb14.get_intermediate_layers(dino_tgt_pose, 1, return_class_token=True)[0]
                dino_tgt_pose = torch.cat((tmp[1].unsqueeze(1), tmp[0]), dim=1)
                tmp = dino_v2_vitg14.get_intermediate_layers(dino_src_img, 1, return_class_token=True)[0]
                dino_src_img = torch.cat((tmp[1].unsqueeze(1), tmp[0]), dim=1)
                tmp=0

            x = torch.randn(args.batch_size, 4, args.image_size // 8, args.image_size // 8).to(device)
            model_kwargs = dict(cfg_scale=args.cfg_scale,
                                dino_tgt_pose=dino_tgt_pose,
                                src_img=src_img,
                                dino_src_img=dino_src_img,
                                target_pose=z_target_pose
                            )



            t = torch.tensor([1.0, 0.0], device="cuda")


            def denoiser(t, x):
                input = torch.cat([x,src_img, z_target_pose], dim=1)
                if args.nfe_count:
                    return model(t, input, **model_kwargs)
                else:
                    return accelerator.unwrap_model(model).forward_with_cfg(t, input, **model_kwargs)

            fake_image = odeint(
                denoiser,
                x,
                t,
                method=args.method,
                atol=args.atol,
                rtol=args.rtol,
                adjoint_method=args.method,
                adjoint_atol=args.atol,
                adjoint_rtol=args.rtol,
                options=options,
                adjoint_params=model.parameters(),
            )

            fake_image=fake_image[-1]
            fake_image = first_stage_model.decode(fake_image/ args.scale_factor).sample

        full_iter += 1
        fake_image = torch.clamp(to_range_0_1(fake_image), 0, 1)

        if not args.nfe_count:
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            if iteration != 0:
                average_time+=curr_time

        else:
            count_nfe=accelerator.unwrap_model(model).nfe
            average_nfe += count_nfe

        for i in range(len(target_path)):
            torchvision.utils.save_image(fake_image[i], os.path.join(args.save_dir, "{}".format(path[i])))




    if args.nfe_count:
        print(f"Average NFE: {average_nfe/full_iter}")
    else:
        print(f"average time: {average_time/(full_iter-1)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator",
        type=str,
        default="determ",
        help="type of seed generator",
        choices=["dummy", "determ", "determ-indiv"],
    )
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
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
    parser.add_argument("--f", type=int, default=8, help="downsample rate of input image by the autoencoder")
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=12, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of image")
    parser.add_argument("--save_dir", default='/home/ubuntu/result/save_imgs', help="storing generated image")

    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")

    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--cfg_scale", type=float, default=1.25, help="Scale for classifier-free guidance")


    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--batch_size", type=int, default=1, help="input batch size")

    parser.add_argument("--n_sample", type=int, default=10000, help="number of sampled images")
    parser.add_argument("--num_steps", type=int, default=40)

    parser.add_argument("--use_karras_samplers", action="store_true", default=False)
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument("--rtol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",
        help="solver_method",
        choices=[
            "dopri5",
            "dopri8",
            "adaptive_heun",
            "bosh3",
            "euler",
            "midpoint",
            "rk4",
            "heun",
            "multistep",
            "stochastic",
            "dpm",
        ],
    )
    parser.add_argument("--data_dir", default="/home/ubuntu/dataset/deepfashion")
    parser.add_argument("--step_size", type=float, default=0.01, help="step_size")
    parser.add_argument("--perturb", action="store_true", default=False)
    parser.add_argument("--nfe_count", action="store_true", default=False)

    parser.add_argument("--num_proc_node", type=int, default=1, help="The number of nodes in multi node env.")
    parser.add_argument("--num_process_per_node", type=int, default=2, help="number of gpus")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int,  help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")
    parser.add_argument("--master_port", type=str, default="23456", help="port for master")

    args = parser.parse_args()
    sample_and_test(args)

