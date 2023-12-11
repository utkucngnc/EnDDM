diffusion_config = {
    "sampler": "ddim",
    "steps": 1000,
    "noise_schedule": "linear",
    "model_mean_type": "epsilon",
    "model_var_type": "learned_range",
    "dynamic_threshold": False,
    "clip_denoised": True,
    "rescale_timesteps": False,
    "timestep_respacing": 100
}

model_config = {
    "image_size": 256,
    "num_channels": 256,
    "num_res_blocks": 2,
    "channel_mult": "",
    "learn_sigma": True,
    "class_cond": False,
    "use_checkpoint": False,
    "attention_resolutions": "32,16,8",
    "num_heads": 4,
    "num_head_channels": 64,
    "num_heads_upsample": -1,
    "use_scale_shift_norm": True,
    "dropout": 0.0,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
    "model_path": "models/pre_trained/256x256_diffusion_uncond.pt"
}