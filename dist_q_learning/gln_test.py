from typing import Callable, List, Text, Tuple

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from gated_linear_networks import gaussian
import haiku as hk

import glns

def _get_dataset(input_size, batch_size=None):
  """Get mock dataset."""
  if batch_size:
    inputs = jnp.ones([batch_size, input_size, 2])
    side_info = jnp.ones([batch_size, input_size])
    targets = 0.8 * jnp.ones([batch_size])
  else:
    inputs = jnp.ones([input_size, 2])
    side_info = jnp.ones([input_size])
    targets = jnp.ones([])

  return inputs, side_info, targets


def main(unused_argv):

    gln1 = glns.GGLN([4,5,6,1], input_size=5,
            context_dim=4, batch_size=1)

    X_input, X_train, Y_train = _get_dataset(5, batch_size=1)

    Y_train= Y_train.item()
    preds= gln1.predict(X_train)

    print(preds)

    gln1.predict(X_train, target=Y_train)

    preds= gln1.predict(X_train)

    print(preds)


if __name__ == '__main__':
    main(1)