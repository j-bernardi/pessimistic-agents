import sys
import argparse
import env
import tensorflow as tf
import numpy as np

class MyParser(argparse.ArgumentParser):
    
    def error(self, message):
        sys.stderr.write('error: {}\n'.format(message))
        self.print_help()
        sys.exit(2)


def smooth_over(list_to_smooth, smooth_last):
    smoothed = [list_to_smooth[0]]
    for i in range(1, len(list_to_smooth)+1):
        if i < smooth_last:
            smoothed.append(
                sum(list_to_smooth[:i]) / len(list_to_smooth[:i]))
        else:
            assert smooth_last == len(list_to_smooth[i-smooth_last:i])
            smoothed.append(
                sum(list_to_smooth[i-smooth_last:i]) / smooth_last
                )
    return smoothed


def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            return func
        return dec(func)
    return decorator


def get_batch_from_memory(memory, batch_size):

    minibatch_i = np.random.choice(
        len(memory),
        min(batch_size, len(memory)),
    )
        
    minibatch = [memory[i] for i in minibatch_i]

    args_as_tuple = tuple(map(tf.convert_to_tensor, zip(*minibatch)))

    return args_as_tuple


class EnvTracker():
    """
    A class that can preserve a half-run environment
    Recreates the env then saves some left-over information
    """

    def __init__(self, env_wrapper):

        class_name = env_wrapper.__class__.__name__
        callable_class = getattr(env, class_name)
        if hasattr(env_wrapper, "kwargs"):
            self.env_wrapper = callable_class(env_wrapper.kwargs)
        else:
            self.env_wrapper = callable_class()

        self.env = self.env_wrapper.env
        self.latest_state = self.env.reset()
        self.return_so_far = 0.
        self.steps_so_far = 0


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        return tf.squeeze(
            tf.random.categorical(logits, 1),
            axis=-1)