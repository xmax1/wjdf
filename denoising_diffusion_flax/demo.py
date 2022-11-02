



import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import jax
from utils import make_grid
from PIL import Image
from train import * 
from flax.training import common_utils
from flax import jax_utils
import wandb
import utils
from sampling import sample_loop, ddpm_sample_step
from configs import oxford102
from pathlib import Path
from utils import to_wandb_config

root_directory = 'denoising-diffusion-flax'
example_directory = 'denoising-diffusion-flax/denoising_diffusion_flax'
repo, branch =  'https://github.com/yiyixuxu/denoising-diffusion-flax', 'main'

rng = jax.random.PRNGKey(0)
cfg = oxford102.get_config()
train_iter = train.get_dataset(rng, cfg)

workdir = '.'
sample_dir = Path(workdir, "samples")

cfg.ddpm.p2_loss_weight_gamma = 1
p2_weights = utils.get_ddpm_params(cfg.ddpm)['p2_loss_weight']

ema_decay_fn = create_ema_decay_schedule(cfg.ema)

wandb.init(
    entity=cfg.wandb.entity,
    project=cfg.wandb.project,
    job_type=cfg.wandb.job_type,
    cfg=to_wandb_config(cfg)
)
    # set default x-axis as 'train/step'
    #wandb.define_metric("*", step_metric="train/step")

num_steps = cfg.training.num_train_steps

rng, state_rng = jax.random.split(rng)

state = create_train_state(state_rng, cfg)

loss_fn = get_loss_fn(cfg)

ddpm_params = utils.get_ddpm_params(cfg.ddpm)
ema_decay_fn = create_ema_decay_schedule(cfg.ema)

train_step = functools.partial(
    p_loss, 
    ddpm_params=ddpm_params, 
    loss_fn =loss_fn, 
    self_condition=cfg.ddpm.self_condition, 
    is_pred_x0=cfg.ddpm.pred_x0, 
    pmap_axis ='batch'
)

p_train_step = jax.pmap(train_step, axis_name = 'batch')
p_apply_ema = jax.pmap(apply_ema_decay, in_axes=(0, None), axis_name = 'batch')
p_copy_params_to_ema = jax.pmap(copy_params_to_ema, axis_name='batch')

train_metrics = []
hooks = []

sample_step = functools.partial(
    ddpm_sample_step, 
    ddpm_params=ddpm_params, 
    self_condition=cfg.ddpm.self_condition, 
    is_pred_x0=cfg.ddpm.pred_x0
)

p_sample_step = jax.pmap(sample_step, axis_name='batch')


for step, batch in zip(tqdm(range(1, n_steps)), train_iter):
    rng, *train_step_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    train_step_rng = jnp.asarray(train_step_rng)
    state, metrics = p_train_step(train_step_rng, state, batch)
    
    for h in hooks:
        h(step)

    # update state.params_ema
    if (step + 1) <= cfg.ema.update_after_step:
        state = p_copy_params_to_ema(state)

    elif (step + 1) % cfg.ema.update_every == 0:
        ema_decay = ema_decay_fn(step)
        state =  p_apply_ema(state, ema_decay)

    if cfg.training.get('log_every_steps'):
        
        train_metrics.append(metrics)
        
        if (step + 1) % cfg.training.log_every_steps == 0:
            
            train_metrics = common_utils.get_metrics(train_metrics)
            
            summary = {
                f'train/{k}': v
                for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
            }
            
            d, t0 = time() - t0, time() 
            summary['time/seconds_per_step'] = d / cfg.training.log_every_steps

            wandb.log({
                    "train/step": step, 
                    **summary
                })
    
    # Save a checkpoint periodically and generate samples.
    if (step + 1) % cfg.training.save_and_sample_every == 0 or step + 1 == num_steps:

        samples = []
        for i in trange(0, cfg.training.num_sample, cfg.data.batch_size):
            rng, sample_rng = jax.random.split(rng)
            samples.append(sample_loop(sample_rng, state, tuple(batch['image'].shape), p_sample_step, cfg.ddpm.timesteps))
        samples = jnp.concatenate(samples) # num_devices, batch, H, W, C

        grid = make_grid(samples, cfg.training.num_sample, padding=2, pad_value=0.1)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
        ndarr = np.array(ndarr)
        im = Image.fromarray(ndarr)

        # utils.wandb_log_image(samples, step+1)

        ### LOG MODEL
        if step + 1 == num_steps and cfg.wandb.log_model:
            artifact = wandb.Artifact(name=f"model-{wandb.run.id}", type="ddpm_model")
            artifact.add_file( f"{workdir}/checkpoint_{step}")
            wandb.run.log_artifact(artifact)

        ### LARGE FILES

# Wait until computations are done before exiting
jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()