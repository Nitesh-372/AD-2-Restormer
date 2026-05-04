# AD2 Reatformer

Small PyTorch project for paired image restoration on adverse-weather style data. The repository includes:

- a custom dataset loader for paired `input` and `target` images
- a lightweight restoration model with an `AD2` degradation-conditioning branch
- a training script
- a single-image inference script
- a bundled dataset split and pretrained checkpoint

## What It Does

`SimpleRestormer` is a compact encoder-decoder network that predicts a restored image from a degraded input image.

The model has two parts:

- `AD2Module`: extracts global image features and predicts a 2-class degradation distribution
- restoration backbone: encodes the image, conditions the features with the AD2 branch, and decodes the final restored image

The training target is a clean paired image with the same filename as the degraded input.

## Project Structure

```text
.
|-- dataset/
|   |-- full/
|   |-- train/
|   |-- val/
|   `-- test/
|-- dataset.py
|-- model.py
|-- model.pth
|-- split.py
|-- train.py
`-- test.py
```

Important folders:

- `dataset/full/input` and `dataset/full/target`: full paired dataset
- `dataset/train`, `dataset/val`, `dataset/test`: split dataset used by training and evaluation scripts

Current split sizes in this repo:

- train: `14455` image pairs
- val: `1807` image pairs
- test: `1807` image pairs

## Requirements

Install these Python packages:

```bash
pip install torch opencv-python numpy tqdm
```

This project has been exercised with the local Windows setup in this repo, but the code itself is standard Python and should also run on Linux/macOS with the same dependencies.

## How Data Loading Works

`WeatherDataset` in [dataset.py](/d:/project/AD2%20reatformer/dataset.py:1):

- reads matching filenames from the input and target directories
- loads images with OpenCV
- converts BGR to RGB
- resizes every image to `256 x 256`
- converts images to normalized PyTorch tensors in `C x H x W` format

Because filenames are matched directly, both folders must contain the same files.

## Train

Run:

```bash
python train.py
```

What `train.py` currently does:

- checks whether CUDA is available
- loads training data from `dataset/train/input` and `dataset/train/target`
- loads validation data from `dataset/val/input` and `dataset/val/target`
- trains `SimpleRestormer` with:
  - batch size `8`
  - optimizer `Adam`
  - learning rate `1e-4`
  - loss `L1Loss`
- runs for `1` epoch
- saves weights to `model.pth`

Notes:

- training overwrites `model.pth`
- there is no argument parsing yet
- there is no checkpoint versioning or resume support yet

## Inference

Run:

```bash
python test.py
```

What `test.py` currently does:

- loads `model.pth`
- reads the hardcoded input image at `dataset\\full\\input\\city_read_12537.jpg`
- resizes it to `256 x 256`
- runs inference
- saves the restored result as `output.png`

The script currently works as checked in and produces `output.png`.

If you want to test another image, edit this line in [test.py](/d:/project/AD2%20reatformer/test.py:13):

```python
img_path = "dataset\\full\\input\\city_read_12537.jpg"
```

## Create New Train/Val/Test Splits

Run:

```bash
python split.py
```

`split.py`:

- reads filenames from `dataset/full/input`
- shuffles them randomly
- creates an `80% / 10% / 10%` split
- copies matching files from `dataset/full/input` and `dataset/full/target` into:
  - `dataset/train`
  - `dataset/val`
  - `dataset/test`

Important:

- the split is not deterministic because no random seed is set
- rerunning the script will reshuffle and recopy files

## Model Summary

`SimpleRestormer` in [model.py](/d:/project/AD2%20reatformer/model.py:24) includes:

- an `AD2Module` that predicts degradation probabilities
- a small convolutional encoder
- a `1x1` feature projection layer
- multiplicative feature conditioning
- a convolutional decoder that outputs a 3-channel restored image

The forward pass returns:

```python
restored_image, degradation_prediction = model(x)
```

## Limitations

This repository is a straightforward research/prototype-style implementation. A few practical limitations are worth knowing up front:

- image size is fixed to `256 x 256`
- paths are hardcoded in the scripts
- training is fixed to `1` epoch
- there is no CLI, config file, or metrics logging
- the degradation prediction branch is computed, but the training script optimizes only restoration loss

## Suggested Next Improvements

- add command-line arguments for data paths, epochs, batch size, and checkpoint path
- save checkpoints with timestamps or epoch numbers
- add PSNR/SSIM validation metrics
- make dataset splitting reproducible with a fixed seed
- add inference support for arbitrary image paths from the command line

