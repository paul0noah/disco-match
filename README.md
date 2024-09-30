# 🪩 DiscoMatch: Fast Discrete Optimisation for Geometrically Consistent 3D Shape Matching

Official repository of the ECCV 2024 paper DiscoMatch: Fast Discrete Optimisation for Geometrically Consistent 3D Shape Matching by Paul Roetzer`*`, Ahmed Abbas`*`, Dongliang Cao, Florian Bernard and Paul Swoboda.
For more information, please visit our [our project page](https://paulroetzer.github.io/publications/2024-09-30-discomatch.html).

`*`: Authors contributed equally.

## ⚙️ Installation
### Prerequesites
You need a working c++ compiler and cmake.
Note: builds are only tested on unix machines.

### Installation Step-by-Step

1) Create python environment
```bash 
conda create -n disco-match python=3.8 # create new virtual environment
conda activate disco-match
conda install pytorch cudatoolkit -c pytorch # install pytorch
git clone git@github.com:paul0noah/disco-match.git
cd disco-match
pip install -r requirements.txt # install other necessary libraries via pip
```

2) Install sm-comb (code to create the windheuser problem, also includes sm-comb solver)
```bash
git clone git@github.com:paul0noah/sm-comb.git
cd sm-comb
python setup.py install
cd ..
```

3) Install disco match solver
```bash
git clone git@github.com:LPMP/BDD.git
cd BDD
git checkout f377a82736435bc4988e2c41e5c8029c168e9505
python setup.py install
cd ..
```

## 📝 Dataset
Datasets are available from this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link). Put all datasets under `./datasets/` such that the directory looks somehow like this
Two example files for `FAUST_r` shapes are included in this repository.
```bash
├── datasets
    ├── FAUST_r
    ├── SMAL_r
    ├── DT4D_r
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.


### 🧑‍💻️‍ Usage
See `discomatch_example.py` for example usage.

## 🚧 Troubleshooting
### Shapes not readable
There are some issues with the `.off` file format. Use e.g. meshlab to convert them to `.obj` for example
### `torch.cuda_isavailable() == False`
- Torch cuda not availabe:
    The answer containing the `--upgrade --force-reinstall` hint in this stackoverflow (thread)[https://stackoverflow.com/questions/70340812/how-to-install-pytorch-with-cuda-support-with-pip-in-visual-studio] solved it
    ```bash
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall
    ```
### OMP not found (when installing BDD)
- (on macOS) install OpenMP with Homebrew `brew install libomp` → gives you e.g. `/opt/homebrew/Cellar/libomp/`
- add the following to `BDD/setup.py` after the indicated line:
    ```python
    if _cuda_flag == "ON":
        cmake_args.append('-DWITH_CUDA=ON')
    ### ADDED LINES:
    cmake_args.append('-DOpenMP_CXX_FLAGS="-Xclang -fopenmp -I /path/to/libomp/include/"')
    cmake_args.append('-DOpenMP_C_FLAGS="-Xclang -fopenmp -I /path/to/libomp/include/"')
    cmake_args.append('-DOpenMP_CXX_LIB_NAMES=libomp')
    cmake_args.append('-DOpenMP_C_LIB_NAMES=libomp')
    cmake_args.append('-DOpenMP_libomp_LIBRARY=/path/to/libomp/lib/libomp.dylib')
    cmake_args.append('-DCMAKE_SHARED_LINKER_FLAGS="-L /path/to/libomp/lib -lomp -Wl,-rpath, /opt/homebrew/Cellar/libomp/17.0.6/lib"')
    ### END OF ADDED LINES
    ```
    where `/path/to/libomp/` should be replaced with e.g. `/opt/homebrew/Cellar/libomp/17.0.6/`
- clear the build directory and run installation again (i.e. `python setup.py install`)

### Some libs for sm-3dcouple not found
- opengl not found:
`sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev`

- if `libxrandr` or `libxinerama` or other libs not found install them via
```bash
sudo apt-get install libxrandr-dev
sudo apt-get install libxinerama-dev
```
List of potential libs not found: `libxrandr`, `libxinerama`, `libxcursor`, `libxi`

## 🙏 Acknowledgement
The implementation of DiffusionNet is based on [the official implementation](https://github.com/nmwsharp/diffusion-net).
The framework implementation is adapted from [Unsupervised Deep Multi Shape Matching](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching).
This repository is adapted from [Unsupervised-Learning-of-Robust-Spectral-Shape-Matching](https://github.com/dongliangcao/Unsupervised-Learning-of-Robust-Spectral-Shape-Matching).

## 🎓Attribution
```bibtex
@inproceedings{roetzerabbas2024discomatch,
    author     = {Paul Roetzer and Ahmed Abbas and Dongliang Cao and Florian Bernard and Paul Swoboda},
    title     = { DiscoMatch: Fast Discrete Optimisation for Geometrically Consistent 3D Shape Matching },
    booktitle = {In Proceedings of the European conference on computer vision (ECCV)},
    year     = 2024
}
```

## License 🚀
This repo is licensed under MIT licence.
