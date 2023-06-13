# ForecastingNonverbalSignals

  

This is the implementation for the paper [Context-Aware Human Behaviour Forecasting in Dyadic Interactions](https://proceedings.mlr.press/v173/tuyen22a.html).

## Dependencies
* python 3.6
* tensorFlow 1.15
* numpy
* pickle5
* sklearn
* pandas
* h5py

## Usage

1. Install virtual environment named SocialActionGAN with dependencies:

``` console
conda create -n SocialActionGAN python=3.6 tensorflow=1.15 pickle5 scikit-learn pandas h5py
```

2.  Download [UDIVA_2d.pickle](https://drive.google.com/drive/folders/1I3xFvgljFxjImlNdR7qvP9M4AvgEI7Dc?usp=sharing), and put it in the folder [dataset](https://github.com/TuyenNguyenTanViet/ForecastingNonverbalSignals/tree/main/dataset). Training model with the default parameters:

``` console
(SocialActionGAN): python train.py
```

3. Alternatively, download the [pre-trained model](https://drive.google.com/drive/folders/1oohhV4Rryfw09Y1XMsBsS1bQorDx0HaC?usp=sharing) and put it in the folder [model](https://github.com/TuyenNguyenTanViet/ForecastingNonverbalSignals/tree/main/model). Forecast the motions, generate the ouput file based on the format of the challenge:

``` console
(SocialActionGAN): python generate.py --annotations_dir "/path_to/talk_annotations_test_masked/" --segments_path "/path_to/test_segments_topredict.csv"
```

## Optional

1. Extract the training data, package it as UDIVA_2d.pickle:

``` console
(SocialActionGAN): python preprocessing.py --annotations_dir "/path_to/talk_annotations_train"
```

## License
[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Citation
If you use this repository for your research, please cite:

```
@misc{tuyen2021forecasting,
      title={Forecasting Nonverbal Social Signals during Dyadic Interactions with Generative Adversarial Neural Networks}, 
      author={Nguyen Tan Viet Tuyen and Oya Celiktutan},
      year={2021},
      eprint={2110.09378},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}

@misc{tuyen2022context,
  title={Context-Aware Human Behaviour Forecasting in Dyadic Interactions},
  author={Tuyen, Nguyen Tan Viet and Celiktutan, Oya},
  booktitle={Understanding Social Behavior in Dyadic and Small Group Interactions},
  pages={88--106},
  year={2022},
  organization={PMLR}
}
```