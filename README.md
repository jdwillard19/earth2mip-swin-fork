# Earth-2 MIP Fork w/ Swin Transformer V2 implementation
Forked from original repository: [\[Earth2Mip\]](https://github.com/NVIDIA/earth2mip)


# Guide for running swin with earth2mip

`source set_interactive_vars.sh`: this sets the relevant environment variables for pytorch and for earth2mip.  Two important ones are the path to 73var dataset with q and the Earth2mip\_Model\_Registry. This also activates the conda environment that I have been using for running swin.

`srun -N 1 --ntasks-per-node=4 --gpus-per-node=4 -u -n 4 python -m earth2mip.inference_ensemble config.json`: that is the run script that I use on the interactive queue.  It can be changed for a job script accordingly.

Note: one thing that is in progress: I used to be able to run with more than 1 node.  I've been having some problems as of late: I think it's because of the modulus DistributedManager and SLURM-clusters.  Perhaps I have to do something with export\_DDP\_vars.sh

# Creating a model registry for Swin

The idea for a model registry to have everything necessary to run the model for inference.  This includes the weights, the global\_means.npy, etc.  There is another important file called `metadata.json` that specifies a loader function.  The loader function determines which loading method that earth2mip should use to load the model.  Loader function are specified in `earth2mip/loaders.py` for Swin.

There are 2 swin versions that we have been using: `earth2mip/networks/swin/` and `earth2mip/networks/swin_residual/`.  The relevant loader loads the appropriate one based on the loader specified in the metada.json for the model.



<!-- markdownlint-disable -->
[![Project Status: Active - The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![GitHub](https://img.shields.io/github/license/NVIDIA/earth2mip)](https://github.com/NVIDIA/earth2mip/blob/master/LICENSE.txt)
[![Documentstion](https://img.shields.io/website?up_message=online&up_color=green&down_message=down&down_color=red&url=https%3A%2F%2Fnvidia.github.io%2Fearth2mip%2F&label=docs)](https://nvidia.github.io/earth2mip/)
[![codecov](https://codecov.io/gh/NickGeneva/earth2mip/graph/badge.svg?token=0PDBMHCH2C)](https://codecov.io/gh/NickGeneva/earth2mip/tree/main)
[![Python versionm: 3.10, 3.11, 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue
)](https://github.com/NVIDIA/earth2mip)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- markdownlint-enable -->

Earth-2 Model Intercomparison Project (MIP) is a Python based AI framework that
enables climate researchers and scientists to explore and experiment with the use of AI
models for weather and climate.
It provides reference workflows for understanding how AI models capture the physics of
the Earth's atmosphere and how they can work with traditional numerical weather
forecasting models.
For instance, the repo provides a uniform interface for running inference using
pre-trained model checkpoints and scoring the skill of such models using certain
standard metrics.
This repository is meant to facilitate the weather and climate community to come up with
good reference baseline of events to test the models against and to use with a variety
of data sources.

## Installation

Earth-2 MIP will be installable on PyPi upon general release.
In the mean time, one can install from source:

```bash
git clone git@github.com:jdwillard19/earth2mip-swin-fork.git

cd earth2mip-swin-fork && pip install .
```

See [installation documentation in original repository](https://nvidia.github.io/earth2mip/userguide/install.html)
for more details and other options.

## Getting Started

Earth-2 MIP provides a set of examples which can be viewed on the [examples documentation](https://nvidia.github.io/earth2mip/examples/index.html)
page which can be used to get started with various workflows.
These examples can be downloaded both as Jupyer Notebooks and Python scripts.
The source Python scripts can be found in the [examples](./examples/) folders.

### Basic Inference

Earth-2 MIP provides high-level APIs for running inference with AI models.
For example, the following can be used to run Pangu weather using an initial state from
the climate data store (CDS):

```bash
python
>>> import datetime
>>> from earth2mip.networks import get_model
>>> from earth2mip.initial_conditions import cds
>>> from earth2mip.inference_ensemble import run_basic_inference
>>> time_loop  = get_model("e2mip://dlwp", device="cuda:0")
>>> data_source = cds.DataSource(time_loop.in_channel_names)
>>> ds = run_basic_inference(time_loop, n=10, data_source=data_source, time=datetime.datetime(2018, 1, 1))
>>> ds.chunk()
<xarray.DataArray (time: 11, history: 1, channel: 69, lat: 721, lon: 1440)>
dask.array<xarray-<this-array>, shape=(11, 1, 69, 721, 1440), dtype=float32, chunksize=(11, 1, 69, 721, 1440), chunktype=numpy.ndarray>
Coordinates:
  * lon      (lon) float32 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
  * lat      (lat) float32 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
  * time     (time) datetime64[ns] 2018-01-01 ... 2018-01-03T12:00:00
  * channel  (channel) <U5 'z1000' 'z925' 'z850' 'z700' ... 'u10m' 'v10m' 't2m'
Dimensions without coordinates: history
```

And you can get ACC/RMSE like this:
```
>>> from earth2mip.inference_medium_range import score_deterministic
>>> import numpy as np
>>> scores = score_deterministic(time_loop,
    data_source=data_source,
    n=10,
    initial_times=[datetime.datetime(2018, 1, 1)],
    # fill in zeros for time-mean, will typically be grabbed from data.
    time_mean=np.zeros((7, 721, 1440))
)
>>> scores
<xarray.Dataset>
Dimensions:        (lead_time: 11, channel: 7, initial_time: 1)
Coordinates:
  * lead_time      (lead_time) timedelta64[ns] 0 days 00:00:00 ... 5 days 00:...
  * channel        (channel) <U5 't850' 'z1000' 'z700' ... 'z300' 'tcwv' 't2m'
Dimensions without coordinates: initial_time
Data variables:
    acc            (lead_time, channel) float64 1.0 1.0 1.0 ... 0.9686 0.9999
    rmse           (lead_time, channel) float64 0.0 2.469e-05 0.0 ... 7.07 2.998
    initial_times  (initial_time) datetime64[ns] 2018-01-01
>>> scores.rmse.sel(channel='z500')
<xarray.DataArray 'rmse' (lead_time: 11)>
array([  0.        , 150.83014446, 212.07880612, 304.98592282,
       381.36510987, 453.31516952, 506.01464974, 537.11092269,
       564.79603347, 557.22871627, 586.44691243])
Coordinates:
  * lead_time  (lead_time) timedelta64[ns] 0 days 00:00:00 ... 5 days 00:00:00
    channel    <U5 'z500'
```
### Deterministic Scoring Example
Deterministic scoring of a swin model can be done using the following workflow. 
```
from earth2mip import model_registry
from earth2mip.inference_medium_range import score_deterministic

registry = model_registry.ModelRegistry('/pscratch/sd/j/jwillard/FCN_exp/earth2mip_model_registry/')
config_path = './config_swin.json'
output_path = './outputs/output_folder/'
h5_folder = "/pscratch/sd/p/pharring/73var-6hourly/staging/"
time_mean = np.load('/pscratch/sd/p/pharring/73var-6hourly/staging/stats/time_means.npy')
with open(config_path) as f:
    config = json.load(f)
registry = model_registry.ModelRegistry('/pscratch/sd/j/jwillard/FCN_exp/earth2mip_model_registry/')
model = get_model(config['weather_model'], registry, device='cuda:0')
datasource = hdf5.DataSource.from_path(
    root=h5_folder, channel_names=model.channel_names
)

scores_swin = score_deterministic(model,
    data_source=datasource,
    n=28, #number of lead times
    initial_times=initial_times,
    output_directory=output_path,
    time_mean=time_mean)

```
### Generating Forecasts Example
Once you've created a registry folder and a folder with ERA5 data, you can generate and save a set of forecasts by doing the following workflow. This can be parallelized for better efficiency if more than one GPU is available. 

```
from earth2mip import model_registry
registry = model_registry.ModelRegistry('/pscratch/sd/j/jwillard/FCN_exp/earth2mip_model_registry/')
model = model_registry.get_model(config_tmp['weather_model'], registry, device=device)
time = datetime.datetime(2018, 1, 1, 0)
initial_times = [time + datetime.timedelta(hours=12 * i) for i in range(730)]

datasource = hdf5.DataSource.from_path(
    root=h5_folder, channel_names=model.channel_names
)

time_mean = np.load('/pscratch/sd/p/pharring/73var-6hourly/staging/stats/time_means.npy')
config_path = './config_swin.json'
output_path = './outputs/output_folder/'
with open(config_path) as f:
    config = json.load(f)
config = EnsembleRun.parse_obj(config)
run_over_initial_times(time_loop=model, data_source=datasource, 
                    initial_times=initial_times, 
                    config=config, output_path=output_path, 
                    shard=1,n_shards=1, score=False)
```

### Additional Resources

- [Earth-2 Website](https://www.nvidia.com/en-us/high-performance-computing/earth-2/)
- [NVIDIA Modulus](https://github.com/NVIDIA/modulus)
