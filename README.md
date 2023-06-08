# Hate_speech_detection
Tuple: Stores duplicates, can store multiple values like int, str.                

Hate Speech Detection


Aim:
Main focus of this project is to find out whether the emotion expressed by the user in his/her text is hateful or not.
Why is it important?
It helps in sorting the reviews given by the public for any particular product which can help in maintaining the company’s or brand’s reputation. It can help understand the behaviour of the public, preferences, and develop targeted advertising campaigns.
can help political campaigns understand public opinion and tailor their messaging accordingly.
What are the initial steps?
Loading the libraries
Loading the datasets from Google Drive
Finding out whether our dataset contains any null value or not?
And as the code processed the dataset does not contain any NAN or null value. 
After this, train_data[train_data['label'] == 0].head(15)
This code line is responsible for finding out the tweets which have the label values=0, And similarly for label==1 the code is written.


train_data['label'].value_counts().plot.bar(colour = 'green' , figsize=(4,4))

This line is generating a histogram on the basis of counts of label value.
Generated a column of length of tweets,
train_data.groupby('label').describe()

Grouping the training data by the label column which represents the max,mean,min and quartiles of id’s associated with the label that is length and id.
Countvectorizer is a tool that is used to deal with the texts to convert it into the arithmetic forms like in this code, we used it and converted the tweets into matrix representations.
cv.vocabulary_.items() retrieves the items (word, index) from the vocabulary of the CountVectorizer object (cv). The vocabulary is a dictionary where the keys are unique words in the corpus, and the values are their corresponding indices.
Fit_transform is used to convert the tweet to 
stop_words = 'english'
This is used for removing words like the, is, a which do not provide specific meanings.
This code snippet uses the vocabulary of the CountVectorizer object to iterate over the unique words and their corresponding indices. For each word, it retrieves the frequency of occurrences from the sum_words matrix and creates a tuple of the word and its frequency. 
These tuples are then appended to the words_freq list, resulting in a list of word-frequency pairs that can be further analysed or visualised.
After that we calculated the sum of unique words and plotted the histogram for it in a descending order.
Word Cloud gets the maximum used word in the largest icon and least used word in the smallest icon.
After this we printed the same for these negative values.

//Read this carefully
Defining a function by importing library regular expressions,

A Regular Expressions (RegEx) is a special sequence of characters that uses a search pattern to find a string or set of strings. It can detect the presence or absence of a text by matching it with a particular pattern, and also can split a pattern into one or more sub-patterns.
r character (r’portal’) stands for raw, not regex. The raw string is slightly different from a regular string, it won’t interpret the \ character as an escape character. This is because the regular expression engine uses \ character for its own escaping purpose.
It contains many other characters as well such as dot(.), $,(),[],^.
For understanding the meaning of each refer gfg’s article
nltk.freqDist:
Used for recording the frequency of the event that has occurred. Means frequency distribution of the experiment.
Tokenization is the start of the NLP process, converting sentences into understandable bits of data that a program can work with.
We have used the nltk.freqDist to plot the histogram between the regular words and frequency of it. And similarly for hashtags_negative.
Tokenization is a simple process that takes raw data and converts it into a useful data string.
Tokenization is used in natural language processing to split paragraphs and sentences into smaller units that can be more easily assigned meaning.
.apply() allow the users to pass a function and apply it on every single value of the Pandas series.
Lambda function is basically a function that can be used to create a single line expressions function. 
Lambda x:y, here x is the bound variable, and y is the body. 
x takes the argument, evaluates and returns the value y. 
.split() is used for splitting.
In the lambda function if the axis is 0 then it is applied to all the columns and if it is 1 then all then all the rows.
After tokenizing, we are creating a word to vector model  using geneism library mentioning all the requirements, Now in word2vec model
1st parameter is window size, that is number of words after or before your target variable (recollect the video from yt)
The 2nd parameter is mean_count which means there should be a certain minimum number of words in a sentence. 3rd is worker which means how many threads you need to use. 

model_w2v.wv.most_similar(positive = "dinner")
model_w2v is assumed to be an instance of a trained Word2Vec model.
wv stands for "word vectors" and is a property of the Word2Vec model that provides access to the trained word vectors.
most_similar() is a method of the wv property that returns the most similar words to a given word based on the learned word embeddings.
The positive parameter is set to "dinner", indicating that we want to find words similar to "dinner". Note that the positive parameter can accept a list of words if you want to find words similar to multiple words.
The method returns a list of tuples, where each tuple contains a word and its similarity score. The similarity score represents how similar the word is to "dinner" based on the trained word embeddings. The higher the score, the more similar the word is.

Train_corpus:
Preprocessing of the text data to remove all the symbols converting it to the lower case we are using nltk re
It initialises a PorterStemmer object from the nltk library. The Porter stemming algorithm is a common technique used to reduce words to their base or root form (e.g., "running" to "run", "cats" to "cat").
It applies stemming to each word in the tweet, but only if the word is not in the set of English stop words provided by the nltk library. Stopwords are commonly used words (e.g., "and'', "the'', "is") that are often removed from text data as they usually do not carry significant meaning.
It joins the stemmed words back together into a single string, separated by spaces.
It appends the processed tweet to the train_corpus list.
By the end of the loop, train_corpus will contain a list of preprocessed tweets where each tweet has undergone lowercase conversion, removal of non-alphabetic characters, word tokenization, stemming, removal of stopwords, and rejoining into a string format. This preprocessed data can then be used for further analysis or training a machine learning model.

At the end applying different models such as random forest, logistic regression and Decision tree classifier to the test train data and predicting the accuracies for the three.

Thank You!





