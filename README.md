# Emma: Emotional Multidimensional Analysis
#### Sentiment Analysis Validation Dataset for Danish
Emma is a validation dataset for Danish sentiment analysis tools consisting of (currently) 352 domain-general sentences rated by 30 raters on four emotional dimensions. This will gradually increase in quality and size.
- [Introduction](#introduction)
- [Citation](#citing-this-work)
- [Emma](#emma)
- [Sentida V2](#sentida-v2)
- [Descriptions of resources](#descriptions-of-folders)
- [Installation](#installation)
- [Examples](#examples)
### Introduction
#### Context
Created by Søren Orm and Esben Kran.
> Emma: Danish Computational Analysis of Emotion in Text
> (by S. Orm and E. Kran)

For questions and commercial use, please contact:
* Esben Kran C.
  * contact@esbenkc.com
  * Aarhus University, [CINeMa](https://inema.webflow.io)
* Søren Orm H.
  * sorenorm@live.dk
  * Aarhus University, [CINeMa](https://inema.webflow.io)

#### Emma
A new domain general validation dataset called Emma that exceeds current validation methods in both complexity and quality, created with a proprietary advanced, adaptive tool for supervised machine learning data collection utilizing a citizen science approach.  It consists of a large amount of coders with a wide demographic representation and sentences scored in a four-dimensional emotional circumplex space that allows for future multidimensional, fine-grained machine learning based Danish sentiment analysis (SA).

If you are Danish, you can help improve the tool by rating the sentences or sharing the form (it updates with new sentences for every trial): https://forms.gle/rhGmE8QZRQpp74WNA

#### Sentida V2
Additionally, this repository has the new state-of-the-art Danish sentiment analysis tool upgraded from the previous state-of-the-art Sentida to V2. Sentida V2 shows significant improvement in classifying sentiment in text compared to Sentida (p < 0.01) in three different validation datasets (TP, TP2, Emma).

Built from the previous iteration of state-of-the-art Danish SA, [Sentida](https://github.com/guscode/sentida) and programmed from the [VADER](https://github.com/cjhutto/vaderSentiment) sentiment analysis python implementation.

### Citing this work
If you use either the dataset or any of the Sentida V2 sentiment analysis tool in your research, please cite the above paper.

### Descriptions of folders
All data is in .csv format in UTF-8 encoding.
* Emma
    Includes the Emma validation dataset with all scorings given by coders as well as all 584 texts scored.
* Sentida V2
    Includes the python implementation as well as aarup.csv and intensifier.csv that are from Sentida (Lauridsen, Dalsgaard, & Svendsen, 2019). The python implementation is built from the previous iteration of state-of-the-art Danish SA, [Sentida](https://github.com/guscode/sentida) and programmed from the [VADER](https://github.com/cjhutto/vaderSentiment) sentiment analysis python implementation.
* Tests
    Includes the thousands of logistic tests performed for each tool (V2, V1, AFINN, and uPS) on each dataset (TP, TP2, Emma) and a summary of the results.
* Validation datasets
    Includes TP.csv and TP2.csv used in the paper and previous studies (Lauridsen et al., 2019)

### Installation
You can install SentidaV2 through pip with the following command:
```
pip install sentida
```
### Documentation and examples
See documentation in the SentidaV2 subfolder.
```
from sentida.sentida2 import Sentida2, sentida2_examples
s2 = Sentida2()
s2.sentida2("Programmering er dejligt. Jeg elsker sentimentanalyse!!", output = "by_sentence_mean", normal = True, speed = "fast")
```

### References
Lauridsen, G. A., Dalsgaard, J. A., & Svendsen, L. K. B. (2019). SENTIDA: A New Tool for Sentiment Analysis in Danish. Journal of Language Works - Sprogvidenskabeligt Studentertidsskrift, 4(1), 38–53.

Hutto, C. J., & Gilbert, E. (2014, May 16). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Eighth International AAAI Conference on Weblogs and Social Media. Eighth International AAAI Conference on Weblogs and Social Media. https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109
