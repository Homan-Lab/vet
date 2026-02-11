# Variance Estimation Toolkit (VET)

Disclaimer: This is not an officially supported Google product.

## Overview

This repository contains code for generating simulated item/response distributions of various shapes, and measuring the p-value of comparisons between those distributions.  Its primary use case is facilitating the power analysis of human ratings for comparing two versions of an AI system.
It has three main components:

   1. `parameterized_sample.py` Generates a large (typically 1000) number of
   samples of simulated output from three sources: a
   pool of human annotators and two machines, using known probability
   distributions.
   2. `response_resampler.py` Estimates p-values directly on the data generated
   by `parameterized_sample.py` and also via resampling on the first sampled set
   of data from `parameterized_sample.py` It then compares the results on the
   generated data and resampled data, where the resampled estimates are the
   kinds of estimates available to investigators under normal experiment
   conditions, and the estimates from `parameterized_sample.py` are a more
   accurate value of the p-value under a null hypothesis based on the actual
   distributions that generated the data. `response_resampler.py` can be run
   in parallel, to save time.

## Usage

### Example usage for response sample generation

```shell
python parameterized_sample.py --exp_dir=/data_dir/path --distortion=.02
```

Where:

`--exp_dir` is the file path where the experiment input and output data are located.

`--generator` is the type of random sample generator. Right now it supports   either normal distribution generator (`ALT_DISTR_GEN`) or Likert normal distribution generator (`TOXICITY_DISTR_GEN`). A Likert normal distribution here refers to a normal distribution with the probability value thresholded to the closest (equispaced) interval corresponding to the Likert scale normalized between 0 and 1.

`--distortion` is a floating point number controlling the amount of distribution generation error in the
second machine sample distribution relative to the human sample distribution.

`--n_items` is the number of items per response set. Each item is assumed to have its own distribution.

`--k_responses` is the number of responses per item. They can be annotators responses for human results or machine responses for machine results.

`--num_samples` is the number of sample sets per experiment. Each set contains `n_items * k_responses` samples for human, machine1 and machine2 results.

`--use_pickle` decides whether to save the sample data in pickle format. When set to false, the samples are saved in json format, which is more readable but less efficient in storage space.

### Example usage for computing metrics over the generated samples

```shell
python response_resampler.py --exp_dir=/path/to/experiment/ --input_response_file=input_file_prefix --config_file=config.csv --line_num=45
```

Where:

`--exp_dir` is the path to the input/output files for the experiment.

`--input_response_file` is the name of the input file. The output file name is generated based on the input file name and the experiment config.

`--line_num` is the number of config line to run. When running in parallel, each resampler job can do the processing according to one line of the experiment config.

`--config_file` is the name of the config csv file located in `<exp_dir>/config/<config_file>`.

`--n_items` and `--k_responses` are similar to the flags in `parameterized_sample.py`. They can be redefined for resampling.

## If you use VET, please cite our work

```shell
@inproceedings{wein-etal-2023-follow,
    title = "Follow the leader(board) with confidence: Estimating p-values from a single test set with item and response variance",
    author = "Wein, Shira  and
      Homan, Christopher  and
      Aroyo, Lora  and
      Welty, Chris",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.196/",
    doi = "10.18653/v1/2023.findings-acl.196",
    pages = "3138--3161",
    abstract = "Among the problems with leaderboard culture in NLP has been the widespread lack of confidence estimation in reported results. In this work, we present a framework and simulator for estimating p-values for comparisons between the results of two systems, in order to understand the confidence that one is actually better (i.e. ranked higher) than the other. What has made this difficult in the past is that each system must itself be evaluated by comparison to a gold standard. We define a null hypothesis that each system{'}s metric scores are drawn from the same distribution, using variance found naturally (though rarely reported) in test set items and individual labels on an item (responses) to produce the metric distributions. We create a test set that evenly mixes the responses of the two systems under the assumption the null hypothesis is true. Exploring how to best estimate the true p-value from a single test set under different metrics, tests, and sampling methods, we find that the presence of response variance (from multiple raters or multiple model versions) has a profound impact on p-value estimates for model comparison, and that choice of metric and sampling method is critical to providing statistical guarantees on model comparisons."
}
```

```shell
@article{homan2024many,
  title={How Many Ratings per Item are Necessary for Reliable Significance Testing?},
  author={Homan, Christopher and Korn, Flip and Pandita, Deepak and Welty, Chris},
  journal={arXiv preprint arXiv:2412.02968},
  year={2024}
}
```

```shell
@article{pandita2025forest,
  title={Forest vs Tree: The $(N, K) $ Trade-off in Reproducible ML Evaluation},
  author={Pandita, Deepak and Korn, Flip and Welty, Chris and Homan, Christopher M},
  journal={arXiv preprint arXiv:2508.03663},
  year={2025}
}
```
