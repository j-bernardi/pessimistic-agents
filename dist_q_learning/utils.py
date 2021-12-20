import math
import os

import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import torch as tc
from tests.check_gpu import check_gpu

try:
    import torch_xla.core.xla_model as xm
except:
    xm = None

try:
    from google.cloud import storage
except ImportError:
    storage = None
    print("Cloud storage package not detected!")


def get_device(device_id=0):

    if xm is not None:
        return xm.xla_device(device_id)
    else:
        return tc.cuda.current_device()

def set_gpu():

    torch_gpu_available = check_gpu()
    dev_i = None
    if torch_gpu_available and tc.cuda.device_count() > 1:
        dev_i = int(input("Input device number (or 'cpu'): "))
        tc.cuda.device(dev_i)
    return dev_i


def geometric_sum(r_val, gamm, steps):
    # Two valid ways to specify infinite steps
    if steps is None or steps == "inf":
        return r_val / (1. - gamm)
    else:
        return r_val * (1. - gamm ** steps) / (1. - gamm)


def get_beta_plot(alpha, beta, n_samples):
    """Returns xs and f (as in f(xs)) needed to plot the beta curve"""
    xs = np.linspace(0., 1., num=n_samples)
    ps = scipy.stats.beta(alpha, beta).pdf(xs)
    return xs, ps


def plot_beta(a, b, show=True, n_samples=10000):
    """Plot a beta distribution, given these parameters."""
    ax = plt.gca()
    xs, ys = get_beta_plot(a, b, n_samples)

    ax.set_title(f"Beta distribution for alpha={a}, beta={b}")
    ax.set_ylabel("PDF")
    ax.set_xlabel("E(reward)")
    ax.set_xlim((0, 1))

    ax.plot(xs, ys)
    if show:
        plt.show()

    return ax


def stack_batch(batch, lib=np):
    """Return a stack"""
    # Default axis is 0
    return tuple(lib.stack(x) for x in zip(*batch))


vec_stack_batch = jax.jit(lambda x: stack_batch(x, lib=jnp))


def jnp_batch_apply(f, x, bs):
    """Apply f to x in fixed batch sizes, then stack at the end

    Args:
        f (Callable): function to apply
        x (jnp.ndarray): apply f to x. Shape (N, ...)
        bs (int): batch size
    Returns:
        Array shape (N, ...) transformed by f
    """
    if hasattr(x, "shape"):
        x_shape = x.shape[0]
        individual_shape = x.shape[1:]
    else:
        x_shape = len(x)
        if hasattr(x[0], "shape"):
            individual_shape = x[0].shape
        else:
            raise NotImplementedError()

    n_batches = math.ceil(x_shape / bs)

    # First, pad x so that it's a multiple of 64
    num_padding = (n_batches * bs) % x_shape
    if num_padding:
        pad = jnp.zeros((num_padding,) + individual_shape)
        padded_x = jnp.concatenate([x, pad], axis=0)
    else:
        padded_x = x
    split_x = jnp.stack(jnp.split(padded_x, n_batches))
    applied_batch = jax.vmap(f)(split_x)
    stacked_result = jnp.concatenate(applied_batch, axis=0)
    if num_padding:
        return stacked_result[:-num_padding]
    else:
        return stacked_result


class JaxRandom:
    """Singleton for jax random numbers

    Ensures that a
    """

    _instance = None

    def __new__(cls, device_id=0):
        if cls._instance is None:
            print("Creating new instance of jax random key")
            cls.key = jax.random.PRNGKey(
                jax.device_put(0, jax.devices()[device_id]))
            cls._instance = super(JaxRandom, cls).__new__(cls)
        return cls._instance

    def update_key(self):
        """Split and update the internal key"""
        key, subkey = jax.random.split(self.key)
        self.key = subkey

    def uniform(self, *args, **kwargs):
        rand_nums = jax.random.uniform(self.key, *args, **kwargs)
        self.update_key()
        return rand_nums

    def choice(self, *args, **kwargs):
        choices = jax.random.choice(self.key, *args, **kwargs)
        self.update_key()
        return choices

    def randint(self, *args, **kwargs):
        rand = jax.random.randint(self.key, *args, **kwargs)
        self.update_key()
        return rand


def upload_blob(source_file_name, destination_blob_name, overwrite=False):
    """Uploads a file to the bucket

    Adds suffix to the filename until it doesn't exist on the cloud

    Args:
        source_file_name (str): The path to your file to upload
        destination_blob_name (str): storage object name to go to the
            cloud
        overwrite (bool): overwrite, or increment file name
    """
    if storage is None:
        print(f"Cannot store on cloud: {source_file_name}")
        return
    bucket_name = os.environ["BUCKET_NAME"]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    def exists(f):
        cloud_file = storage.Blob(bucket=bucket, name=f)
        return cloud_file.exists(storage_client)

    counter = 0
    destination_blob_name_with_suffix = destination_blob_name
    while exists(destination_blob_name_with_suffix) and not overwrite:
        counter += 1
        if counter > 100:
            raise FileExistsError("Likely recursive write - stop!")
        split_path = destination_blob_name.split(".")
        split_path[-2] = split_path[-2] + f"_{counter}"
        destination_blob_name_with_suffix = ".".join(split_path)
    if counter > 0:
        print(
            f"File {destination_blob_name} exists {counter} times! "
            f"Added suffix")
    elif exists(destination_blob_name_with_suffix):
        assert overwrite, "This was unexpected."
        print("File exists - overwriting")

    blob = bucket.blob(destination_blob_name_with_suffix)
    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to bucket {bucket_name}, file: "
          f"{destination_blob_name_with_suffix}")


def download_blob(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket

    Args:
        source_blob_name (str): the name of the file on the cloud
        destination_file_name (str): path to which to download the file
            to
    """
    if storage is None:
        print(f"Cannot fetch from cloud: {source_blob_name}")
        return
    # The ID of your GCS bucket
    bucket_name = os.environ["BUCKET_NAME"]

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        f"Downloaded storage object {source_blob_name} from bucket "
        f"{bucket_name} to local file {destination_file_name}.")


def device_put_id(x, device_id):
    if hasattr(x, "device") and x.device() == device_id:
        return x
    else:
        return jax.device_put(x, device=jax.devices()[device_id])
