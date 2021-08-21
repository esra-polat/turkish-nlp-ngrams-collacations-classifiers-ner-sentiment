# Turkish NLP 
These programs was implemented with Python on the Kaggle Notebook.

---------
# n-grams
How we clean corpus from meaningless characters and stop words:
```
vtokenized = [idx.lower() for idx in words.split() 
if idx.lower() not in stopwords.words('turkish') and re.findall("^[a-zA-Z0-9ğüşöçİĞÜŞÖÇ]+$", idx) and len(idx) > 1]
```

We import stopwords, nltk and re library:
```
import nltk, re
from nltk.corpus import stopwords
nltk.download('stopwords')
```

We has a method that finds words that have the characters we write as regex: 

```
re.findall
```

How ve find ngrams:

```
unigrams = ngrams(tokenized, 1)
unigramsFreq = collections.Counter(unigrams)
df = pd.DataFrame(unigramsFreq.most_common(1000), columns = ['Words', 'Score'])
```

We used library named ngrams from nltk.util:

```
from nltk.util import ngrams
```

This method takes two parameters, first one is the corpus we want to find ngrams and the second one is how many words this ngram takes. For example 1 means unigram, 2 means bigram etc.
We applied this method for unigrams, bigrams and trigram. After that we give the result to Counter to find ngrams scores.

![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%231-n-grams/1.png)
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%231-n-grams/2.png)
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%231-n-grams/3.png)

---------

# Collocations

#### Method 1: Pointwise Mutual Information
Pointwise mutual information (PMI), or point mutual information, is a measure of association used in information theory and statistics. In contrast to mutual information (MI) which builds upon PMI, it refers to single events, whereas MI refers to the average of all possible events.

The Pointwise Mutual Information (PMI) score for bigrams is: 
```
PMI(w1,w2) = log2 [ P(w1,w2) / ( P(w1)P(w2) ) ]
```
For trigrams:
```
PMI(w1,w2,w3) = log2 [ P(w1,w2,w3) / ( P(w1)P(w2)P(w3) ) ]
```
The main intuition is that it measures how much more likely the words co-occur than if they were independent. However, it is very sensitive to rare combination of words. For example, if a random bigram ‘abc xyz’ appears, and neither ‘abc’ nor ‘xyz’ appeared anywhere else in the text, ‘abc xyz’ will be identified as highly significant bigram when it could just be a random misspelling or a phrase too rare to generalize as a bigram. Therefore, this method is often used with a frequency filter.

#### Method 2: The T-Test

The t-test is any statistical hypothesis test in which the test statistic follows a Student's t-distribution under the null hypothesis.
A t-test is the most commonly applied when the test statistic would follow a normal distribution if the value of a scaling term in the test statistic were known. When the scaling term is unknown and is replaced by an estimate based on the data, the test statistics (under certain conditions) follow a Student's t distribution. The t-test can be used, for example, to determine if the means of two sets of data are significantly different from each other.

#### Method 3: Chi-Square Test
A chi-squared test, also written as χ2 test, is a statistical hypothesis test that is valid to perform when the test statistic is chi-squared distributed under the null hypothesis, specifically Pearson's chi-squared test and variants thereof. Pearson's chi-squared test is used to determine whether there is a statistically significant difference between the expected frequencies and the observed frequencies in one or more categories of a contingency table.

![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%232-Collocations/collacations-results/1.png)
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%232-Collocations/collacations-results/2.png)

We can see that T-test method give very good results. These results are also very similar with our n-grams results. Frequency and T-test methods are also similar. When we compare the T-test results with n-gram results, we can say that PMI and Chi-Square methods did not give successful results. We can also run different tests to see which list makes the most sense for a particular dataset. Alternatively, we can combine results from multiple lists.  We find it effective to multiply T-test and frequency, taking into account both the probability increase and the frequency of its occurrence.

---------

# Deep Learning Classifiers

We used the Kickstarter Campaigns Dataset from Kaggle*
The dataset have 20632 entries of 67 features.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/1.png)

We examined and cleaned the dataset.
We detected and dropped missing values and we dropped unnecessary columns.
Finally, we converted the values to 0 and 1.
Left hand side in this slide, we can see that campaign states and right hand sites campaigns success or failure rates. 
Only %29 of campaigns were successful in this dataset.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/2.png)

When we look at the general statistics, we see how each feature covers a very different range.
We observed the distributions of several columns.
We did this to get an idea about data.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/3.png)

We examined outlier data.
For example, trim backers_count, pledged and create_to_launch_days then create a new IQR dataframe with these truncated values.
This reduction resulted in a dataframe where there are 9308 instances, with only the IQR for the variables in question remaining.
Correlations between each variable against SuccessfulBool, which remember, is a binary value where 0=failed and 1=succeeded.
Looking at the correlations above we can see that nothing is too strongly correlated except spotlight, backers_count, pledged, and staff_pick.
But really the only significant ones are backers_count and spotlight.
In order to be able to make predictions in the light of all these data, we will combine its features and create our x cluster.
We will add log_goal into the reduced_x_features dataframe and saw log_pledged shows a significant boost as well, so that will be included.

When we transformed goal and pledged,
we found that these had a stronger correlation than their original forms, so these new features were added to reduced_x_feature.

![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/4.png)

X_train, y_train and X_test, y_test values of our data sets.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/5.png)

#### CNN (Convolutional Neural Network)

Convolutional neural network (CNN, or ConvNet) is a class of deep neural network, most commonly applied to analyze visual imagery.[1] They are also known as shift invariant or space invariant artificial neural networks (SIANN), based on the shared-weight architecture of the convolution kernels or filters that slide along input features and provide translation equivariant responses known as feature maps.

##### Prediction of Successful with CNN
We see a few examples of results. There are some correct predictions as well as some incorrect ones here. Accuracy value is seventy-six percent.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/7.png)

#### LSTM (Long Short-Term Memory)
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. The Long Short-Term Memory (LSTM) cell can process data sequentially and keep its hidden state through time.

##### Prediction of Successful with LSTM
When we examine some results, we see that the results obtained are not only 0 and 1, but the accuracy value is higher.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/9.png)

We can see that LSTM is more successful and sensitive in these calculations where we use the success status as the y variable. When we compared the results below, we can say that: When we search the answer of ‘predict if a project/campaign will be successful or not’, the LSTM algorithm works better than the CNN algorithm.
| Model | Accuracy |
| ------------- | ------------- |
| CNN  | 76.91%  |
| LSTM  | 90.66%  |

#### Prediction of the Amount of Money Collected

We have done all the operations we mentioned in the previous slides for the "Pledged" value instead of "Successful State". Therefore, we don’t again explain the same steps. We want to add a few important details about pledged value. We use the pledged as the y variable. Our x variables are the same.

##### Prediction of Amount of Money Collected with CNN
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/10.png)

##### Prediction of Amount of Money Collected with LSTM
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%233-deep-learning-nlp/11.png)

We can see that LSTM is more successful and sensitive in these calculations where we use the pledged as the y variable. When we compared the results below, we can say that: When we search the answer of ‘predict the amount of money collected’, the LSTM algorithm works better than the CNN algorithm. In addition, we observed that LSTM gives more sensitive results.

| Model | Accuracy |
| ------------- | ------------- |
| CNN  | 65.66%  |
| LSTM  | 92.13%  |


---------

# Named Entity Recognition

We used the TBMM Dataset labelled versions. 
We can see the word cloud of the most common 500 word using in the dataset. 
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%234-named-entity-recognition/1.png)

We used the indexes of the elements for the X and Y sets. We used the pad_sequence method to make all the strings in the list have the same length. This method puts 0 at the beginning of each array until it is the same length as the longest array. We created the x set with words and the y set with tags.

#### BI-LSTM (Bidirectional Recurrent Neural Networks)
Bidirectional recurrent neural networks(RNN) are really just putting two independent RNNs together. This structure allows the networks to have both backward and forward information about the sequence at every time step. Using bidirectional will run your inputs in two ways, one from past to future and one from future to past and what differs this approach from unidirectional is that in the LSTM that runs backward you preserve information from the future and using the two hidden states combined you are able in any point in time to preserve information from both past and future.

#### CRF (Conditional Random Fields)
CRF is a discriminant model for sequences data similar to MEMM. It models the dependency between each state and the entire input sequences. Unlike MEMM, CRF overcomes the label bias issue by using global normalizer.
In this article, we are focusing on linear-chain CRF which is a special type of CRF that models the output variables as a sequence. This fits our use case of having sequential inputs. For example, in natural language processing, linear chain CRFs are popular, which implement sequential dependencies in the predictions. In image processing the graph typically connects locations to nearby and/or similar locations to enforce that they receive similar predictions.

Here are the training and validation graphics. Validation accuracy is 96 percent. Looking at these graphs, we can see that the correct values are used. The learning and validation curves look good.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%234-named-entity-recognition/2.png)


We see the samples we got from the test set, their prediction and true values.
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%234-named-entity-recognition/3.png)

---------

# Sentiment Classification
We used the combine of four Turkish datasets.
- Beyaz Perde Comments
- IMDB Comments
- Product Comments 1
- Product Comments 2
They contain usually a text column and a sentiment column. We combined them and created one big dataset. We edited columns again for us.

#### LSTM (Long Short-Term Memory)
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture[1] used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. The Long Short-Term Memory (LSTM) cell can process data sequentially and keep its hidden state through time.

![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%235-sentiment-classification/1.png)
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%235-sentiment-classification/2.png)
![](https://github.com/esra-polat/turkish-nlp-ngrams-collacations-classifiers-ner-sentiment/blob/main/Delivery%235-sentiment-classification/3.png)
