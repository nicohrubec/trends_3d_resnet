# TReNDS Neuroimaging Kaggle Solution

In this repo you can find the code for the 3D Resnet part of our solution for the 2020 TReNDS Neuroimaging Kaggle competition. In the end our team placed 37th out of 1047 participating teams.

## Problem Description
The challenge in this competiton was to predict age and assessment values from two domains using features derived from brain MRI images.
The data consisted of 11754 unknown patients. For half of those we were given labels to train our models and the other half represented the test set.
As data input we got the following information for each patient:
  1. 53 3D spatial maps extracted from fMRI image timecourses --> images of coherent brain regions for a patient
  2. functional connectivity values between brain regions --> how does activity in two parts of the brain correlate or how likely is it that if one part of the brain activates the other activates as well?
  3. loading features --> density of gray matter in certain parts of the brain
  
For more information you can visit the competition website:
https://www.kaggle.com/c/trends-assessment-prediction/overview

## Solution
Our solution consists of two parts.

### 1. Stack of Linear Models
Stacking together a multitude of linear models like Ridge Regressions, Support Vector Machines and Elastic Nets we could reach .1590 loss on the LB. One important thing to note is that the test data consisted of patient data from two different locations or sites. By the competition host we were given some but not all IDs of the site 2 entries in the test set. By looking at the feature distributions you could see quite some shift between the different sites. Just shifting these known site 2 entries to match the statistics of the site 1 entries we got .1588 on the LB.

### 2. Resnet3D
Our first attempts with 3D Resnet all overfitted very heavily due to the high dimensionality of the 3D images and the low number of samples in the training set. It was very hard to get this to learn properly. Hence, we changed the approach and now instead of training on a patient level basis we train on the component level, which means that one sample in our net corresponds to one component image, the matched functional connectivity values and the loading features. We then get predictions for all targets for every component of the patient and the final predictions are just a mean over all component predictions for a patient. For the image path we used a Resnet10 and the tabular features were just added in the head with a simple MLP. This approach alone can get .1599 on the LB. I am sure with more time this approach could be improved even further, for example we did not use any augmentation in the training.

1h prior to the deadline we used our last submission to blend these two solution parts together and this boosted us to public LB .1582/private LB .1588, hence 37th place. With optimized weights this solution could even have reached top 30, unfortunately there was no time left.
