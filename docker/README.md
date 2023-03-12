# PCDet Docker Support
## Preparation
Before building the docker image, make sure to check the
configurations in `config.sh`. 
You may need to change the `HOST_*` variables
to the **absolute** paths of the corresponding directories.
```bash
# In config.sh
HOST_PCDET_ROOT=...     # PCDet location, i.e. the repo itself
HOST_NUSC_ROOT=...      # NuScenes dataset that contains nuscenes-devkit
                        # and data directory (e.g. v1.0-mini)
HOST_CADC_ROOT=...      # CADC dataset directory
HOST_LOGDIR=...         # Output location
                        # can be used to store model checkpoints etc.
```
Also, you need to make sure that the file structure of the dataset 
match the outline in `docs/INSTALL.md`.

## Build Docker Image
Use the provided script to build the docker image.
The name of the image is by default `pcdet-standalone`.
```bash
bash build.sh
```
This process should take a few minutes.

## Run Docker Container
Use the provided script to run the docker container.
The name of the container is by default `pcdet-standalone.$(whoami).$RANDOM`.
```bash
bash run.sh
```
Additional arguments for `docker run` can be passed directly.
For example
```bash
bash run.sh --cpuset-cpus=0,1
```

The following directories from host will be mounted to the container's workspace:
```bash
HOST_PCDET_ROOT  =>  PCDET_ROOT     # `/root/pcdet` by default
HOST_NUSC_ROOT   =>  NUSC_ROOT      # `/root/nusc` by default
HOST_CADC_ROOT   =>  CADC_ROOT      # `/root/cadc` by default
HOST_LOGDIR      =>  LOGDIR         # `/root/logdir` by default
```