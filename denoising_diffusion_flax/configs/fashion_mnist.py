import ml_collections

def get_config():

  config = ml_collections.ConfigDict()

  n_gpu = 1

  # wandb
  config.wandb = wandb = ml_collections.ConfigDict()
  wandb.entity = 'xmax1' # team name, must have already created
  wandb.project = "jax-ddpm-demo"  # required filed if use W&B logging
  wandb.job_type = "training"
  wandb.name = "demo_2" # run name, optional
  wandb.log_train = True # log training metrics 
  wandb.log_sample = True # log generated samples to W&B
  wandb.log_model = True # log final model checkpoint as W&B artifact
  

  # training
  config.training = training = ml_collections.ConfigDict()
  training.num_train_steps = 10000
  training.log_every_steps = 100
  training.loss_type = 'l1'
  training.half_precision = False
  training.save_and_sample_every = 2000
  training.num_sample = 64


  # ema
  config.ema = ema = ml_collections.ConfigDict()
  ema.beta = 0.995
  ema.update_every = 10
  ema.update_after_step = 100
  ema.inv_gamma = 1.0
  ema.power = 2 / 3
  ema.min_value = 0.0
 

  # ddpm 
  config.ddpm = ddpm = ml_collections.ConfigDict()
  ddpm.beta_schedule = 'linear'
  ddpm.timesteps = 1000
  ddpm.p2_loss_weight_gamma = 0. # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
  ddpm.p2_loss_weight_k = 1
  ddpm.self_condition = False # not tested yet
  ddpm.pred_x0 = False # by default, the model will predict noise, if True predict x0


  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'fashion_mnist'
  data.batch_size = 32 * n_gpu
  data.cache = False
  data.image_size = 28
  data.channels = 1


  # model
  config.model = model = ml_collections.ConfigDict()
  model.dim = 32
  model.dim_mults = (1, 2, 4)


  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.optimizer = 'Adam'
  optim.lr = 1e-3
  optim.beta1 = 0.9
  optim.beta2 = 0.99
  optim.eps = 1e-8

  config.seed = 42

  return config


