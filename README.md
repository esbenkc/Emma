# Emma: Emotional Multidimensional Analysis

Emma is a validation dataset for Danish sentiment analysis tools consisting of (currently) 352 domain-general sentences rated by 30 raters on four emotional dimensions. This will gradually increase in quality and size.

- [Emma: Emotional Multidimensional Analysis](#emma-emotional-multidimensional-analysis)
  - [Introduction](#introduction)
    - [Emma](#emma)
  - [Authors](#authors)
  - [Using this work](#using-this-work)
  - [Citing this work](#citing-this-work)
  - [Descriptions of folders](#descriptions-of-folders)
  - [References](#references)

## Introduction

### Emma

A new domain general validation dataset called Emma that exceeds current validation methods in both complexity and quality, created with a proprietary advanced, adaptive tool for supervised machine learning data collection utilizing a citizen science approach. It consists of a large amount of coders with a wide demographic representation and sentences scored in a four-dimensional emotional circumplex space that allows for future multidimensional, fine-grained machine learning based Danish sentiment analysis (SA).

If you are Danish, you can help improve the tool by rating the sentences or sharing the form (it updates with new sentences for every trial): https://forms.gle/rhGmE8QZRQpp74WNA.

## Authors

> Emma: Danish Computational Analysis of Emotion in Text, 2020
> (Søren Orm and Esben Kran)

## Using this work

For questions and commercial use, please contact:

- [Esben Kran](https://github.com/esbenkc). E-mail: (contact@esbenkc.com)[mailto:contact@esbenkc.com]
- [Søren Orm](https://github.com/sorenorm). E-mail: (sorenorm@live.dk)[mailto:sorenorm@live.dk]

## Citing this work

If you use either the dataset or any of the Sentida sentiment analysis tool in your research, please cite the above paper.

## Descriptions of folders

All data is in .csv format in UTF-8 encoding. See our Sentida implementation and documentation on the [official Sentida repository](https://github.com/guscode/sentida).

| Folder/file         | Description                                                                                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Emma                | Includes the Emma validation dataset with all scorings given by coders as well as all 584 texts sentences for the dataset.                              |
| Tests               | Includes the thousands of logistic tests performed for each tool (V2, V1, AFINN, and uPS) on each dataset (TP, TP2, Emma) and a summary of the results. |
| Validation datasets | Includes TP.csv and TP2.csv used in the paper and previous studies (Lauridsen et al., 2019)                                                             |
| *emma.csv*	      | The new validation dataset **emma** from this paper.													|


## References

Lauridsen, G. A., Dalsgaard, J. A., & Svendsen, L. K. B. (2019). SENTIDA: A New Tool for Sentiment Analysis in Danish. Journal of Language Works - Sprogvidenskabeligt Studentertidsskrift, 4(1), 38–53.

Hutto, C. J., & Gilbert, E. (2014, May 16). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Eighth International AAAI Conference on Weblogs and Social Media. Eighth International AAAI Conference on Weblogs and Social Media. https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109
