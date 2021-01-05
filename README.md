# Exercise on Data Analysis

This repository contains an analysis of factors influencing compensation for developers based on data from the stackoverflow 2020 survey for an assignment of a data science course on Udacity.

## Necessary Python libraries

This notebook uses:
* [Numpy] version 1.18.5 
* [Pandas] version 1.0.5
* [pycountry_convert] version 0.7.2
* [Seaborn] version 0.11.1
* [matplotlib]
* [itertools]

### Individual functions are required from:

* [Scipy] 
* [pingouin] 

## Repository contents

| Filename | Description|
| -------- | -----------|
| Exploring Data | main file containing code for loading, preprocessing and visualizing data |
| Preprocessing_Functions.py | Functions used in *Exploring Data* for preprocessing and statistical post-hoc tests|
| Figures | Folder containing the figures generated by Exploring Data|


## Motivation

In the course I took, we looked at factors influencing/correlating with salary but the analysis focused on satisfaction or hours of work per week. The aim of the present analysis was to dig deeper into potential 'confounding' / other factors in the dataset by looking at three specific questions:

## Summary of the results

- **Question 1:** is there a difference in total compensation by continent?

Results show a significant difference between all continents except for Asia vs. Africa. This means that continent needs to be taken into account for further analyses, as it can otherwise distort the interpretation of other questions.

- **Question 2:** does compensation in Europe and North America depend on size of the organisation?

Results show that except for small companies up to 100 employees, compensation is indeed higher for larger company sizes. Analyses were calculated separately for North America and Europe based on a Bonferroni/corrected alpha value of 0.025.


- **Question 3:** Do people with a PhD in North America and Europe receive a higher compensation compared to people with another type of university degree or without a university degree?

Results of a mixed-model Analysis of Variance (ANOVA) with *Continent* as between-subjects factor and *Education Level* as within-subjects factor show a significant interaction effect. Post-hoc testing indicate higher compensation in North America but not in Europe. As the number of years people have been coding professionally is highly and positively correlated with compensation, a separate analysis with *Years of professional coding experience* was run, which indeed indicated that the observed effect for PhD might be confounded by the higher number of professional coding experience.

## Acknowledgement

Data analysed in the course of this project is taken from the 2020 Stackoverflow survey. You can find the official published results here: https://insights.stackoverflow.com/survey/2020

Code for running the mixed-model ANOVA is done using Pingouin:
> Vallat, R. (2018). Pingouin: statistics in Python. 
> Journal of Open Source Software, 3(31), 1026, https://doi.org/10.21105/joss.01026


[Numpy]:<https://numpy.org>
[Scipy]:<https://scipy.org>
[Pandas]:<https://pandas.pydata.org/>
[Seaborn]:<https://seaborn.pydata.org/>
[Pingouin]:<https://pingouin-stats.org/>
[scikit learn]:<https://scikit-learn.org/stable/>
[pycountry_convert]:<https://pypi.org/project/pycountry-convert/>
[matplotlib]:<https://matplotlib.org/>
[itertools]:[https://docs.python.org/3/library/itertools.html]