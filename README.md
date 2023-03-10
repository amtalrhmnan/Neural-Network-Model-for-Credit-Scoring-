# Neural-Network-Model-for-Credit-Scoring-
A neural network model for credit scoring, that involves assessing an individual's creditworthiness based on their financial history and other factors, The results showed that the neural network model outperformed traditional models in terms of accuracy and F1 score.

________________________________________________________________________________________
Abstract


This project aimed to investigate the potential of neural network models for credit scoring, a task that involves assessing an individual's creditworthiness based on their financial history and other factors. I used the (Give me some credit) dataset and developed a neural network model for credit scoring and evaluated its performance in comparison to traditional machine learning models. The results showed that the neural network model outperformed traditional models in terms of accuracy and F1 score, indicating that it was able to extract valuable features from the data and make more informed predictions. The model's ability to handle a large amount of data and capture non-linear relationships between features is particularly beneficial for credit scoring. The project also highlighted the importance of considering ethical issues and fair decision-making in any credit scoring project. Overall, the results demonstrate the potential of neural networks for credit scoring and encourage further research to improve the model's performance and generalizability.

Dataset link: https://www.kaggle.com/c/GiveMeSomeCredit
_____________________________________________________________________
#Background


Credit scoring has a long history that dates back to the 1950s, when it was initially created to impartially evaluate borrowers' creditworthiness. At first, credit scores were generated using straightforward statistical models that considered just a few factors, like a person's credit history and debt-to-income ratio.

With the development of more complex statistical models and the advancement of more data sources, the usage of credit scoring has grown both in scope and sophistication over time. With the expansion of the mortgage market in the 1980s and 1990s, the use of credit scores exploded as lenders started utilizing them to assist determine the risk of lending to property buyers.
With the rise of AI and machine learning in recent decades, credit scoring has continued to advance. Nowadays, credit scoring models frequently use a variety of data sources and cutting-edge Artificial Intelligence techniques, such neural networks, to provide more precise and trustworthy predictions about the creditworthiness of borrowers.

_____________________________________________________________________
#Project objective


In my project, I will be using a Multilayer perceptron MLP neural network to make a prediction of creditworthiness by predicting if the person will experience 90 days past due delinquency or worse in the next two years

_____________________________________________________________________
#Project overview


The process of building a neural network credit scoring model involves several steps. First, Gathering and preprocess a large dataset of credit data. This data may include information about the borrower's credit history, income, and other factors that are relevant to their creditworthiness. Next, appropriate neural network architecture must be chosen and train the model using the preprocessed data. This may involve adjusting the model's parameters and fine-tuning it to achieve optimal performance.
Once the model has been trained, it can be tested on a separate dataset to evaluate its accuracy. If the model performs well, it can be deployed in a production environment, where it can be used to make real-time credit decisions.
Overall, a neural network credit scoring model has the potential to greatly improve the efficiency and accuracy of credit decision-making, helping lenders to better manage risk and enabling more borrowers to access credit.

______________________________________________________________________
#Methodology


The task of the project is to predict a person's or company's creditworthiness. Creating a Multilayer Perceptron (MLP) model that can precisely forecast the probability of a borrower defaulting on a loan.
An extremely popular approach in credit scoring systems is using classical machine learning models such as decision trees and logistic regression these models give a good prediction but with the consideration of the sensitivity of loan risks to lenders, building new approaches with higher accuracy and higher ability to learn as the bigger data is a game changer, As shown in the following figure, deep learning models are superior to classical machine learning models.



![Picture21](https://user-images.githubusercontent.com/88631375/215350788-dcf3875f-1b60-4c34-8d17-1ff0677ae150.png)
_____________________________________________________________________
#Multilayer perceptron (MLP)


A Multilayer Perceptron (MLP) is a fully connected multi-layer neural network.

![image](https://user-images.githubusercontent.com/88631375/215350990-fa1b665d-b17d-47fb-bc58-56c94f84aea0.png)



________________________________________________________________________
#Used packages 


![image](https://user-images.githubusercontent.com/88631375/215351132-84f65a60-69d7-4e80-8647-3e5765b73102.png) NumPy 

![image](https://user-images.githubusercontent.com/88631375/215351181-98730bd7-5870-49fa-a166-1fa3c9e85e18.png) Pandas 

![image](https://user-images.githubusercontent.com/88631375/215351247-16aba9ca-aad1-448b-8481-2dcef74fc919.png) scikit-learn

![image](https://user-images.githubusercontent.com/88631375/215351277-fed0cd19-39d3-4f95-9e80-2dce5ee2604e.png) TensorFlow 

![image](https://user-images.githubusercontent.com/88631375/215351300-44267187-ba3a-4530-a3e9-0c433d0ef998.png) Keras 

![image](https://user-images.githubusercontent.com/88631375/215354754-ee6a431e-b4bd-43c0-b300-2cd650983f25.png)  matplotlib
______________________________________________________________________
#Project steps 


Building a neural network credit scoring system typically involves the following steps:


1.Collect and preprocess the data


This involves gathering the data that I will use to train and test my model and performing any necessary preprocessing steps to prepare the data for analysis. This may include tasks such as cleaning the data, imputing missing values, and normalizing numerical features.

Dataset:


the dataset is taken from a competition on Kaggle website as csv file  with size of (150K,11), the purpose of this Featured Prediction Competition is to build a model to Improve on the state of the art in credit scoring by predicting the probability that somebody will experience financial distress in the next two years.

The following table illustrate the features in the dataset:

[Data Dictionary.xls](https://github.com/amtalrhmnan/Neural-Network-Model-for-Credit-Scoring-/files/10530669/Data.Dictionary.xls)


The following is a more detailed explanation of some of the features of the dataset 


a)personal lines of credit


A personal line of credit is a form of loan that allows you to withdraw funds as needed for a specified length of time. You can borrow up to a predetermined amount, and you can pay it back through checks or bank transfers. When you borrow money, you instantly start paying interest, and you keep doing so until the loan has been repaid. The cap goes from $1,000 to $100,000. You frequently make small monthly payments with a credit card.
Mostly, these loans are not secured by any form of property, so if you are unable to pay them back, the lender cannot seize any of your belongings. Nevertheless, lenders might agree to let you use collateral in return for a lower interest rate. The cost of having this loan also includes yearly or monthly maintenance fees, additional penalties for late or returned payments, and other expenditures. You can use a personal line of credit for everything you require if you don't go over your credit limit. For home equity loans and business expenses, there are other comparable loans that need security or are limited to certain expenses.


b)age


Age can be an important factor in credit scoring because it is often used as an indicator of credit experience and stability. Generally, older borrowers are considered more likely to have a longer credit history and a more stable financial situation, which can be viewed positively by lenders. Additionally, older borrowers are more likely to have established a good credit history and to have a lower risk of default. On the other hand, younger borrowers may be viewed as having less credit experience and a higher risk of default, which can negatively impact their credit score.


c)Number of dependents


The number of dependents of a borrower is considered an important factor in credit scoring because it can indicate the borrower's financial stability. People with more dependents may have higher living expenses and financial obligations, which may impact their ability to repay the loan. They may also have a lower income and less disposable income, which may make it difficult for them to repay the loan. This is not always the case, as a borrower with more dependents may have a higher income and a stable job that will help them repay the loan. As a result, credit scoring systems consider this factor as a critical characteristic in determining a borrower's creditworthiness.

2 Split the data into training and testing sets.

3 Build the model.


This involves designing and implementing the neural network model using keras with TensorFlow library in the back end , The model is then trained on the training data using an optimization algorithm I used stochastic gradient descent (SGD) to adjust the model's parameters and minimize the error between the predicted and actual labels.

![image](https://user-images.githubusercontent.com/88631375/215351899-75532258-d643-4b84-a512-a7154a1219c2.png)

Defining the architecture of the model #


In this step, I have defined the different layers in my model and the connections between them. Since Keras has two main types of models: Sequential and Functional models. I have chosen the Sequential model, then all I have done is defined each layer in the model, an input layer with the size of (None, 10) (the length of the X values) and the output layer is a single neuron layer since it is a binary classification problem, the three hidden layers are set to 100-125-100 respectively, there is no thump rule for the number of neurons in the hidden layers but by trying I found that this is the most stable structure. The following table shows the model architecture

_________________________________________________________________
Layer (type),                 Output Shape,              Param number   
=================================================================
dense (Dense),                (None, 100),               1100      
_________________________________________________________________
dense_1 (Dense),              (None, 125),               12625     
_________________________________________________________________
dense_2 (Dense),              (None, 100),               12600     
_________________________________________________________________
dense_3 (Dense),              (None, 1),                 101       
=================================================================
Total params: 26,426

Trainable params: 26,426

Non-trainable params: 0
_________________________________________________________________

Compile the network #

Compiling code means converting it into a format that a machine can understand. The model.compile() method in Keras performs this function. To compile the model, I define the loss function, which calculates our model???s losses, the optimizer, which reduces the loss, and the metrics, which are used to determine our model???s accuracy.





Fit the network #

Fitting the model to the data after compiling with model.fit(). This is used to train the model on the data. 
Passing the independent and dependent features for training set for training the model as follows:
???	 validation data will be evaluated at the end of each epoch
???	setting the epochs as 100
???	storing the trained model in model_history variable which will be used to visualize the training process
ModelCheckpoint callback is used in conjunction with model.fit() training to store a model or weights (in a checkpoint file) at specified interval so that the model or weights may be loaded later to continue training from the saved state.

Evaluate the model #
Once the model is trained, it is tested on the testing data to evaluate its performance. This involves comparing the model???s predictions to the actual labels and calculating metrics such as accuracy. Using the following code shown in the figure 

![image](https://user-images.githubusercontent.com/88631375/215353642-b0f4f90f-79ec-4b0e-9d36-14e023fb92c0.png)

The Final Accuracy= 0.935366690158844


Make Predictions #


 Using model.predict() to make predictions using my model on test data.

_________________________________________________________________________
#Results and analysis

This project offers a deep learning model using multilayer perceptron to make prediction depending on gathered information about the person who applied for a credit -like his credit history, debit ratio, Monthly Income, age and other features- whither will experience 90 days past due delinquency or worse or not, if the answer is yes (prediction= 1) then it is a ???bad credit??? and it should be rejected, but if the answer is no (prediction =0) then it can be considered as a ???good credit??? and should be accepted by lenders.

The confusion matrix and metrics


Interpreting The Confusion Matrix 


![image](https://user-images.githubusercontent.com/88631375/215353912-1a190825-ca9b-4ada-9a05-a0e26b7ff6c0.png)

	Accuracy

The accuracy is 0.93536667.




	Precision

The precision is 0.94005818


	Recall

The recall is 0.99402654



	F1-Score

The F1-score is 0.9662894




Visualizing the model performance
Using the matplotlib library I can plot the accuracy and the loss for both training and validation data as shown in the following figures.

![image](https://user-images.githubusercontent.com/88631375/215353991-2c131c88-30b2-47d2-8c45-fc05f6318459.png)


![image](https://user-images.githubusercontent.com/88631375/215353997-84820f66-918d-4e84-afc5-6b000b3a4218.png)

____________________________________________________________________
#Understanding the results 


By considering the F1 score, I can say that the performance of my model is excellent and will perform well to predict the target credit scoring, and by looking at the two figures in the previous section there is some sort of instability in the validation data that can be referred to the Biased nature of the data since only Around 6% of samples defaulted (target value =1) and the rest has not defaulted- around 96% of the data -which is a big source of bias in data training leading to a slight instability in predictions, this can be solved by exploring more sensitive neural network architecture, or even using transformers neural networks such as Mixture Of Experts (MOE) this may be possible only with big size of data since these  architectures will not perform well with a relatively small dataset like the one used in this project,  
nevertheless, there is a necessary need for improvement by collecting more data, I have already sent a formal request to the Central Bank of Jordan with a proof letter from the HTU University requesting data to perform more training on the model, furthermore, I have tried to contact CRIF Jordan to provide me with more information about a The Scoring Assessment System, but unfortunately, there was no respond from them.

________________________________________________________________________
#Conclusions and recommendations


In this project, I have developed a neural network model for credit scoring and evaluated its performance using (give me some credit) dataset. The results of the experiments indicate that the neural network model is a promising solution for credit scoring, outperforming traditional machine learning models in terms of accuracy and F1 score, by utilizing a deep learning model, I was able to extract valuable features from the raw data and make more informed predictions than traditional methods.
One of the main strengths of the neural network model is its ability to handle a large amount of data and capture non-linear relationships between features. This is particularly important in credit scoring, where a wide range of factors can influence an individual's creditworthiness.
The results also showed that the model's performance can be improved by fine-tuning the hyperparameters and using more advanced architectures. Additionally, it is important to consider the ethical implications of credit scoring and ensure that the model does not perpetuate biases or discrimination.
In conclusion, the neural network model for credit scoring has the potential to improve the accuracy and efficiency of credit decision-making. I recommend further research to explore other architectures and techniques to improve the model's performance and generalizability to other datasets. Additionally, I recommend considering ethical issues and fair decision-making in any credit scoring project.





