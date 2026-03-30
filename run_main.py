import subprocess
import os 


cmd = [
    "python", "main.py",

    "log_samples=True",        # save generated samples
    "compute_fid=False",        # evaluate FID score
    "log_rec=True",            # log reconstructions
    "compute_rec_fid=False",    # FID on reconstructions

    "dataset=ukb", 
    "dataset.name=ukb",
    "dataset.num_workers=3",   # dataloader workers
    "dataset.batch_size=16",
    'dataset.img_resolution=256',

    "model=covae", # covae
    "project=covae-ukb", # covae-mnist

    "model.total_training_steps=400000",
    "model.step_schedule=exp",
    "model.start_scales=2",
    "model.end_scales=256",

    "model.sigma_min=0.05",
    "model.sigma_max=3",
    "model.time_scale=karras",
    "model.rho=7",

    "model.rec_weight_mode=linear",
    "model.kl_weight_mode=square",

    "model.denoiser_loss_mode=l2",
    "model.loss_mode=huber",
    "model.lambda_denoiser=0.1",
    "model.use_gan=False",

    "network=autoencoder",
    "network.attn_resolutions=[]",
    "network.z_channels=8",
    "network.model_channels=64",
    "network.channel_mult_enc=[1,2,2,4]",
    "network.channel_mult_dec=[1,2,2,4]",

    "gradient_clip_val=200",
    "deterministic=True",

    "dataset.out_channels=2",
]

subprocess.run(cmd)

# command = (
#     "python main.py "
#     "log_samples=True "
#     "compute_fid=True "
#     "log_rec=True "
#     "compute_rec_fid=True "
#     "dataset=mnist "
#     "dataset.name=mnist "
#     "model=covae "
#     "project=covae-mnist "
#     "dataset.num_workers=6 "
#     "dataset.batch_size=32 "
#     "model.total_training_steps=400000 "
#     "model.step_schedule=exp "
#     "model.start_scales=2 "
#     "model.end_scales=256 "
#     "model.sigma_min=0.05 "
#     "model.sigma_max=3 "
#     "model.time_scale=karras "
#     "model.rho=7 "
#     "network=autoencoder "
#     "gradient_clip_val=200 "
#     "model.rec_weight_mode=linear "
#     "model.kl_weight_mode=square "
#     "network.attn_resolutions=[14] "
#     "deterministic=True "
#     "network.z_channels=1 "
#     "network.model_channels=64 "
#     "network.channel_mult_enc=[2,2,2] "
#     "network.channel_mult_dec=[2,2,2] "
#     "model.denoiser_loss_mode=l2 "
#     "dataset.out_channels=2 "
#     "model.loss_mode=huber "
#     "model.lambda_denoiser=0.1 "
#     "model.use_gan=False"
# )


# os.system(command)

