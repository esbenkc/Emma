#### Sentida V2
The new state-of-the-art Danish sentiment analysis tool upgraded from the previous state-of-the-art Sentida to V2. Sentida V2 shows significant improvement in classifying sentiment in text compared to Sentida (p < 0.01) in three different validation datasets (TP, TP2, Emma).

Implementation of the previous state-of-the-art Danish SA in R, [Sentida](https://github.com/guscode/sentida), for python along with SentidaV2 and programmed loosely from the [VADER](https://github.com/cjhutto/vaderSentiment) sentiment analysis python implementation.

- [Authors and Citation](#authors-and-citation)
- [Installation](#installation)
- [Documentation](#documentation-and-examples)
- [Context](#acknowledgments-and-context)
- [Notes on the current state of the program](#notes)
- [References](#references)

### Authors and Citation
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

### Installation
You can install SentidaV2 through pip with the following command:
```
pip install sentida
```
### Documentation and examples
The function:
```
from sentida import Sentida
Sentida().sentida(
                    text,
                    output = ["mean", "total", "by_sentence_mean", "by_sentence_total"],
                    normal = True,
                    speed = ["normal", "fast"]
                    )
# Speed parameter does not have an effect in version <0.2.1
```
WARNING: Setting speed to fast drastically reduces sentiment precision in complex sentences but speeds up the process by 180% (10,000 iteration test).

Usage examples:
```
# Define the class:
SV = Sentida()
_____________________________

SV.sentida(
        text = 'Lad der blive fred.',
        output = 'mean',
        normal = False)

Example of usage:
Lad der bliver fred
Sentiment =  2.0
_____________________________

SV.sentida(
        text = 'Lad der blive fred!',
        output = 'mean',
        normal = False)

With exclamation mark:
Lad der blive fred!
Sentiment =  3.13713
_____________________________

SV.sentida(
        text = 'Lad der blive fred!!!',
        output = 'mean',
        normal = False)

With several exclamation mark:
Lad der blive fred!!!
Sentiment =  3.7896530399999997
_____________________________

SV.sentida(
        text = 'Lad der BLIVE FRED',
        output = 'mean',
        normal = False)

Uppercase:
lad der BLIVE FRED
Sentiment =  3.466
_____________________________

SV.sentida(
        text = 'Det går dårligt.',
        output = 'mean',
        normal = False)

Negative sentence:
Det går dårligt
Sentiment =  -1.8333333333333335
_____________________________

SV.sentida(
        text = 'Det går ikke dårligt.',
        output = 'mean',
        normal = False)

Negation in sentence:
Det går ikke dårligt
Sentiment =  1.8333333333333335
_____________________________

SV.sentida(
        text = 'Lad der blive fred, men det går dårligt.',
        output = 'mean',
        normal = False)

'Men' ('but'):
Lad der blive fred, men det går dårligt
Sentiment =  -1.5
_____________________________

SV.sentida(
        text = 'Lad der blive fred.',
        output = 'mean',
        normal = True)

Normalized:
Lad der blive fred
Sentiment =  0.4
_____________________________

SV.sentida(
        text = 'Lad der bliver fred. Det går dårligt!',
        output = 'by_sentence_mean',
        normal = False)

Multiple sentences mean:
Lad der bliver fred. Det går dårligt!
Sentiments = [2.0, -2.8757025]
_____________________________

SV.sentida(
        text = 'Lad der bliver fred. Det går dårligt!',
        output = 'by_sentence_total',
        normal = False)

Multiple sentences total:
Lad der bliver fred. Det går dårligt!
Sentiments = [2.0, -5.751405]
_____________________________
```
### Acknowledgements and Context
Thanks to CINeMa (https://inema.webflow.io),
the Sentida team, jry, VADER, AFINN, and last
but not least Formula T., for inspiration and encouragement.
For license information, see LICENSE.TXT

The SentidaV2 sentiment analysis tool is freely available for
research purposes (please cite). If you want to use the tool
for commercial purposes, please contact:
    - contact@esbenkc.com
    - sorenorm@live.dk
Or the SentidaV1 team:
    - gustavaarup0111@gmail.com
    - jacdals@hotmail.com
    - larskjartanbachersvendsen@gmail.com

SENTIDA v2.
Aarhus University, Cognitive Science.
2019 - Cognition & Communication.
@authors: sorenorm & esbenkc.

This script was developed along with other tools in an attempt to improve
danish sentiment analysis. The tool will be updated as more data is collected
and new methods for more optimally accessing sentiment is developed.

### Notes
VADER BASIS VALUES

Multiplication values:
    0.291, 0.215, and 0.208 for !, !!, and !!! respectively
        empirically tested by one sentence compared to the three conditions
    0.733 for uppercase
        empirically tested from single control sentence to uppercase version
    0.293 for degree modifications from adverbs
        empirically tested with "extremely"


SENTIDA V2 BASIS VALUES

Currently using VADER basis values
Question mark is: XXX
Degree modifications for other words are implemented in intensitifer list
    - Need implementation of larger intensifier list based on sentences


FUTURE IMPROVEMENTS

Still missing: common phrases, adjusted values for exclamation marks,
Adjusted values for men-sentences, adjusted values for uppercase,
More rated words, more intensifiers/mitigators, better solution than snowball stemmer,
Synonym/antonym dictionary.
Social media orientated: emoticons, using multiple letters - i.e. suuuuuper.
Normalization with respect to sub-(-1) and super-(1) output values

### References
Lauridsen, G. A., Dalsgaard, J. A., & Svendsen, L. K. B. (2019). SENTIDA: A New Tool for Sentiment Analysis in Danish. Journal of Language Works - Sprogvidenskabeligt Studentertidsskrift, 4(1), 38–53.

Hutto, C. J., & Gilbert, E. (2014, May 16). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text. Eighth International AAAI Conference on Weblogs and Social Media. Eighth International AAAI Conference on Weblogs and Social Media. https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109