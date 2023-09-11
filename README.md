# Constrained self-supervised method with temporal ensembling for fiber bundle detection on anatomic tracing data

## Contents
 - [citation](#citation)
 - [installation](#installation)
 - [dependencies](#dependencies)
 - [example](#Examples)
 - [network architecture](#Network-architecture)
 - [results](#Fiber-bundle-segmentation-results)

## Citation

If you use Fiber Segmentation Tool (FibSeg), please cite the following papers:

- Sundaresan, V., Lehman, J.F., Fitzgibbon, S., Jbabdi, S., Haber, S.N. and Yendiki, A., 2022, September. Constrained self-supervised method with temporal ensembling for fiber bundle detection on anatomic tracing data. In International Workshop on Medical Optical Imaging and Virtual Microscopy Image Analysis (pp. 115-125). Cham: Springer Nature Switzerland. Preprint: arXiv:2208.03569

---
---

## Dependencies
- Main fibseg dependencies:
  - Python > 3.6
  - PyTorch=1.5.0
- Extra dependencies for label extraction:
  - TIRL and Slider (https://git.fmrib.ox.ac.uk/seanf/slider)

---
---

## Installation
To install the fibseg tool do the following:
1. Clone the git repository into your local directory.
  - If you are not familiar with GitHub then the easiest way is to use the button labelled **<> Code** (right hand side, just above the file list) on the [main truenet page](/) and select the **Download ZIP** option. After you've done this, move the zip file to where you want to have truenet installed and unzip it.
3. Open up a terminal, go to the directory where you unzipped the file, and then run:
```
python setup.py install
```
3. Use the instructions in this document ([simple usage](#simple-usage) is recommended for beginners)
4. For more advanced usage, detailed lists of options for the subcommands are available in the command-line help:
```
fibseg --help
```
---
--- 

## Examples

 - Run a pretrained model on an histological slice.

`mkdir Results/`

`fibseg evaluate -i Input/directory -m /model_directory/pretrained_model_name.pth -o Results/`

# Note: The link for teh repository containing the pretrained model weights to be updated soon.

---
---

## Network architecture:
<img
src="images/fig2.png"
alt="Network architecture used for fiber segmentation."
/>


## Fiber bundle segmentation results:
<img
src="images/fig4_super_comp.jpg"
alt="Two sample results of the ablation study, with the profile of fibers within detected bundles."
/>


