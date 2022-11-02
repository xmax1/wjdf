# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

import jax
import train
import utils
from configs import oxford102 as config_lib

my_config = config_lib.get_config()
work_dir = './flowers'

my_config.ddpm.p2_loss_weight_gamma = 1
p2_weights = utils.get_ddpm_params(my_config.ddpm)['p2_loss_weight']
ema_decay_fn = train.create_ema_decay_schedule(my_config.ema)
state = train.train(my_config, work_dir)