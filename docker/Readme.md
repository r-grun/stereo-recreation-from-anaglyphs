# Docker

> This directory contains the Dockerfile to build and run the training in a Docker container.
> Currently, only training the U-Net model is supported.

Run all steps from the root directory of the project
`/stereo-recreation-from-anaglyphs`.

## Build the Docker image

To build the Docker image, run the following command from the root directory of the project `/stereo-recreation-from-anaglyphs`:

```bash
docker build -t anaglyph-recreation -f docker/unet.Dockerfile .
```

## Run the Docker container

To run the Docker container with exposed port 6006, mapped data directory of the host machine, and attached GPU support, run the following command:

```bash
docker run --rm -it --runtime=nvidia --gpus '"device=0"' -p 6006:6006 -v <data-dir of the hostmachine>:/data/ --name anaglyph-recreation-container --shm-size=3gb anaglyph-recreation
```

## Start a Jupyter notebook in the container

To run the scripts in the browser, start a Jupyter notebook in the container. Alternatively, run the scripts directly in the terminal.

When using the Jupyter notebook, make sure to run the container with the exposed port `8888`:

```bash
docker run --rm -it --runtime=nvidia --gpus '"device=0"' -p 6006:6006 -p 8888:8888 -v <data-dir of the hostmachine>:/data/ --name anaglyph-recreation-container --shm-size=3gb anaglyph-recreation
```

Once the container is running, start the jupyter notebook inside the container using `docker exec ...`, then open a browser and navigate to [http://localhost:8888/](http://localhost:8888/) to access the Jupyter notebook.

## Tensorboard

During training, the progress and loss values will be displayed in the console as well as in Tensorboard.
Open Tensorboard with the following command from the `image_colorization/` directory.
`--logdir=logs` specifies the directory where the logs are stored.
`--host=0.0.0.0` specifies the host to listen on (use `0.0.0.0` when running in docker container).

```bash
tensorboard --logdir=logs --host=0.0.0.0
```

Once the TensorBoard server has started, navigate to [http://localhost:6006/](http://localhost:6006/) in your browser.