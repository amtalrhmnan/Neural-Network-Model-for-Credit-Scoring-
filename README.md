# Neural-Network-Model-for-Credit-Scoring-
A neural network model for credit scoring, that involves assessing an individual's creditworthiness based on their financial history and other factors, The results showed that the neural network model outperformed traditional models in terms of accuracy and F1 score.

________________________________________________________________________________________
This project aimed to investigate the potential of neural network models for credit scoring, a task that involves assessing an individual's creditworthiness based on their financial history and other factors. I used the (Give me some credit) dataset and developed a neural network model for credit scoring and evaluated its performance in comparison to traditional machine learning models. The results showed that the neural network model outperformed traditional models in terms of accuracy and F1 score, indicating that it was able to extract valuable features from the data and make more informed predictions. The model's ability to handle a large amount of data and capture non-linear relationships between features is particularly beneficial for credit scoring. The project also highlighted the importance of considering ethical issues and fair decision-making in any credit scoring project. Overall, the results demonstrate the potential of neural networks for credit scoring and encourage further research to improve the model's performance and generalizability.

Dataset link: https://www.kaggle.com/c/GiveMeSomeCredit
_____________________________________________________________________
#Background


Credit scoring has a long history that dates back to the 1950s, when it was initially created to impartially evaluate borrowers' creditworthiness. At first, credit scores were generated using straightforward statistical models that considered just a few factors, like a person's credit history and debt-to-income ratio.

With the development of more complex statistical models and the advancement of more data sources, the usage of credit scoring has grown both in scope and sophistication over time. With the expansion of the mortgage market in the 1980s and 1990s, the use of credit scores exploded as lenders started utilizing them to assist determine the risk of lending to property buyers.
With the rise of AI and machine learning in recent decades, credit scoring has continued to advance. Nowadays, credit scoring models frequently use a variety of data sources and cutting-edge Artificial Intelligence techniques, such neural networks, to provide more precise and trustworthy predictions about the creditworthiness of borrowers.
A credit scoring system that incorporates artificial intelligence (AI) and neural networks enables more precise and effective risk assessment. A deep learning system known as a neural network, which takes its structural cues from the human brain, is particularly effective at interpreting complicated and multivariate data. A model that can precisely predict a borrower's creditworthiness can be developed by training a neural network on a large collection of credit data. This is accomplished by giving the neural network inputs, including data on finances and credit histories, and training it to produce a default probability. Artificial intelligence (AI) and neural networks can assist lenders in making more educated and trustworthy lending decisions, ultimately resulting in enhanced loan availability and risk management for borrowers.

A neural network credit score model is a deep learning model used to forecast a person's or company's creditworthiness. Creating a model that can precisely forecast the probability of a borrower defaulting on a loan is the goal of this project. This is a helpful tool for lenders since it enables them to choose which borrowers to lend to and under what conditions.

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

A layer is made up of several neurons stacked in a row, while a multi-layer neural network is made up of many layers that are stacked next to each another.
I have explained the main components of this of structure below.
Input layer
I loaded a CSV file into the input layer to provide the model with the data. It is the only layer that is visible in the entire design of a neural network that transfers all the information from the outside world without any processing, in my project the input layer size will be 10 since I have 10 features in the dataset, the dataset will be explained in the dataset section 

Hidden layers 
Hidden layers are intermediate layers that do all calculations and extract features from data. There can be numerous interconnected hidden layers that account for searching for various hidden features in the data, in my project I built 3 hidden layers with 100-125 –100 neurons with respect to the order. 
Output layer 
The output layer uses data from earlier hidden layers and the model's learnings to make a final prediction. The layer where we obtain the end outcome is the most crucial one. Classification/regression models typically have a single node in the output layer. However, it depends entirely on the nature of the problem at hand and how the model was created, In the case of my project the output layer is a single neuron since it is a binary classification problem 

________________________________________________________________________
#Used packages 


![image](https://user-images.githubusercontent.com/88631375/215351132-84f65a60-69d7-4e80-8647-3e5765b73102.png) NumPy 

![image](https://user-images.githubusercontent.com/88631375/215351181-98730bd7-5870-49fa-a166-1fa3c9e85e18.png) Pandas 

![image](https://user-images.githubusercontent.com/88631375/215351247-16aba9ca-aad1-448b-8481-2dcef74fc919.png) scikit-learn

![image](https://user-images.githubusercontent.com/88631375/215351277-fed0cd19-39d3-4f95-9e80-2dce5ee2604e.png) TensorFlow 

![image](https://user-images.githubusercontent.com/88631375/215351300-44267187-ba3a-4530-a3e9-0c433d0ef998.png) Keras 

![image](https://user-images.githubusercontent.com/88631375/215351336-6183d949-5d2b-4e72-a81a-e401e2349851.png) matplotlib

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

Defining the architecture of the model
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


![image](https://user-images.githubusercontent.com/88631375/215353221-79ca6335-cdc4-4849-8c21-2db57ed505d9.png)



Activation functions 


Relu


The rectified linear unit (ReLU) is a type of activation function commonly used in neural networks, particularly in CNNs and MLPs. It is defined as:


 f(x)=max⁡(0,x),
 
 where x is the input value.

 ReLU is a piece-wise linear function that outputs the input value if it is positive and 0 if it is negative.
f(x)={x ,  x>0   and 0 , x<0  }


 
 ![image](https://user-images.githubusercontent.com/88631375/215353412-61bcb493-88d7-48c1-97e9-aafacab3c94a.png)




ReLU function is not fully interval-derivable, but we can take sub-gradient, as shown in the figure below.



 ![image](https://user-images.githubusercontent.com/88631375/215353425-af2ac0b8-59db-4aa3-b48a-f4e3ff78d643.png)


ReLU is widely used in deep learning because it is computationally efficient and does not require complex mathematical operations like sigmoid or tanh functions. Additionally, ReLU can improve the training speed of a neural network and reduce the risk of the model getting stuck in the saturated state. This function helps the model to converge faster and improve the performance of the model.
ReLU has limitations, in some cases it can produce a problem called “Dying ReLU” where the weights on the negative side of the activation function will update towards zero, resulting in a model that won’t learn from negative inputs. To overcome this problem, Leaky ReLU, PreLU, and RreLU are popular variants of ReLU, but my project is a binary classification model so this will not be a problem.

Sigmoid
A sigmoid function is a common type of activation function in neural networks. It’s a mathematical function that converts any input value to an output value between 0 and 1. The function is named “sigmoid” because it resembles a “S” curve.
 σ(x)=1/((1+e^((-x) ) ) ),
 where x is the input value and e is the mathematical constant (approximately 2.71828). The function accepts any real value as input and returns a value between 0 and 1. As a result, it may be used to describe probability or binary classification issues.
 
 
 ![image](https://user-images.githubusercontent.com/88631375/215353445-d4318f8d-5920-45c6-8957-1dbb29e1d779.png)


The sigmoid function is a type of logistic function, which is popular in deep learning because it has a pleasant mathematical feature that makes it simple to optimize using gradient descent. The sigmoid function and its derivatives are simple to calculate, and it is also a smooth function, making it an excellent candidate for optimization.
The derivative of the sigmoid function is shown in the following figure:

 ![image](https://user-images.githubusercontent.com/88631375/215353464-fad75f48-0256-483f-9df1-02a03cf052ab.png)


2.Compile the network: 

Compiling code means converting it into a format that a machine can understand. The model.compile() method in Keras performs this function. To compile the model, I define the loss function, which calculates our model’s losses, the optimizer, which reduces the loss, and the metrics, which are used to determine our model’s accuracy.
 
![image](https://user-images.githubusercontent.com/88631375/215353520-f6dd9b76-807e-43ae-8457-cb91ba8decea.png)

Binary cross-entropy loss function
Binary cross-entropy loss is a loss function commonly used in binary classification tasks, where the goal is to predict one of two possible outcomes (e.g. true/false, 0/1). The function compares the predicted probability for the true class (labeled as 1) to the actual outcome, and calculates the loss as the negative log-likelihood of the true class. The smaller the loss, the better the model’s prediction. The formula is defined as:

Loss=-(y*log⁡(p)+(1-y)*log⁡(1-p) )

Where:

	y is the true label (0 or 1)
	p is the predicted probability of the positive class
	log is the natural logarithm
What are gradient descent and stochastic gradient descent?
Gradient descent (GD) optimization
The weights are incrementally updated after each epoch (= pass over the training dataset) using the Gradient Decent optimization algorithm.
By taking a step in the opposite direction of the cost gradient, the magnitude and direction of the weight update are computed.
 
Δwj=-η ∂J/∂wj,

where η is the learning rate. The weights are then updated after each epoch by the following update rule:
 
 w≔w+Δw, 
 
where Δw is a vector that contains the weight updates of each weight coefficient w, which are computed as follows:
 
Δwj=-η ∂J/∂wj=-η∑^i▒(t arg⁡e t(i)-output(i))(-x(i)j) =η∑^i▒(t arg⁡e t(i)-output(i))x(i)j.
Gradient Descent optimization can be visualized as a hiker (the weight coefficient) attempting to descend a mountain (cost function) into a valley (cost minimum), with each step determined by the steepness of the slope (gradient) and the hiker’s leg length (learning rate). With a single weight coefficient in a cost function, we can illustrate this concept as follows:
 
 ![image](https://user-images.githubusercontent.com/88631375/215353582-2c452e6e-2cd2-4dbc-b0c3-5fcf29e568fb.png)


Stochastic Gradient Descent
(SGD) is an optimization algorithm that finds the smallest value of a function. It is a gradient descent algorithm that is useful for large-scale optimization problems because it updates model parameters based on a single training example at a time (as opposed to using the entire dataset to compute the gradient). SGD can converge faster and be more computationally efficient than other optimization algorithms because of this. The algorithm begins with a fixed set of parameters and iteratively updates them by moving in the opposite direction of the gradient of the objective function with respect to the parameters. A learning rate determines the step size, which controls how large of a step to take in the opposite direction of the gradient.

3.	Fit the network: 
Fitting the model to the data after compiling with model.fit(). This is used to train the model on the data. 
Passing the independent and dependent features for training set for training the model as follows:
•	 validation data will be evaluated at the end of each epoch
•	setting the epochs as 100
•	storing the trained model in model_history variable which will be used to visualize the training process
ModelCheckpoint callback is used in conjunction with model.fit() training to store a model or weights (in a checkpoint file) at specified interval so that the model or weights may be loaded later to continue training from the saved state.

4.	Evaluate the model: 
Once the model is trained, it is tested on the testing data to evaluate its performance. This involves comparing the model’s predictions to the actual labels and calculating metrics such as accuracy. Using the following code shown in the figure 

![image](https://user-images.githubusercontent.com/88631375/215353642-b0f4f90f-79ec-4b0e-9d36-14e023fb92c0.png)

The Final Accuracy= 0.935366690158844


5.	Make Predictions:
 Using model.predict() to make predictions using my model on test data.

_________________________________________________________________________
#Results and analysis

This project offers a deep learning model using multilayer perceptron to make prediction depending on gathered information about the person who applied for a credit -like his credit history, debit ratio, Monthly Income, age and other features- whither will experience 90 days past due delinquency or worse or not, if the answer is yes (prediction= 1) then it is a “bad credit” and it should be rejected, but if the answer is no (prediction =0) then it can be considered as a “good credit” and should be accepted by lenders.

The confusion matrix and metrics


To make this model useful the results must be understood, one way to do this is using the Confusion Matrix form sklearn.metrics
A confusion matrix is a performance evaluation tool that is particularly useful in assessing recall, precision, specificity, accuracy, and the area under the curve (AUC) of a classification model.
Interpreting The Confusion Matrix 
The terminology of The Confusion Matrix is as follows:
•	True Positives (TP): The model predicted positive, and the actual label is positive
•	True Negative (TN): The model predicted negative, and the actual label is negative
•	False Positive (FP): The model predicted positive, and the actual label was negative
•	False Negative (FN): The model predicted negative, and the actual label was positive
These terms could be presented visually as follows:

![image](https://user-images.githubusercontent.com/88631375/215353864-958b87bb-a307-4388-852b-5021a6b939ca.png)

Now I will perform the Confusion Matrix of the model in similar form 


![image](https://user-images.githubusercontent.com/88631375/215353912-1a190825-ca9b-4ada-9a05-a0e26b7ff6c0.png)

	Accuracy
accuracy is a commonly used metric for evaluating the performance of a model on a given dataset. It is the ratio of the number of correct predictions made by the model to the total number of predictions made. It is often expressed as a percentage. However, it is important to note that accuracy alone may not always be the best metric to evaluate a model's performance, especially when the dataset is imbalanced or when the cost of false positives and false negatives are different.
Accuracy=(TP + TN)/(TP+FP+TN+FN)
The accuracy is 0.93536667.




	Precision
Precision is a metric used to evaluate the performance of a binary classification model. It is defined as the ratio of true positive predictions to the total number of positive predictions. In other words, it is the proportion of correct positive predictions out of all the positive predictions made by the classifier. High precision means that the classifier is good at identifying positive instances, while low precision means that it often makes false positive predictions.
Pr⁡e cision=TP/(TP+FP )
The precision is 0.94005818


	Recall
Recall is a measure of a classifier's ability to correctly identify positive instances from a dataset. It is defined as the ratio of true positive predictions to the total number of actual positive instances. It is often used in the context of binary classification problems, where it measures the proportion of actual positive instances that were correctly identified by the classifier. A high recall value indicates that the classifier has a low false negative rate, meaning that it correctly identifies most of the positive instances.
Recall=TP/(TP+FN)
The recall is 0.99402654



	F1-Score
F1-score is a measure of a model's accuracy and balance between precision and recall. It is the harmonic mean of precision and recall, where the best value is 1.0 and the worst value is 0.0. It is often used as a single number summary of the performance of a classification model, as it takes into account both the false positives and false negatives.
The F1-score is calculated using the following formula:
F1=2*((precision*recall))/((precision+recall) )
F1-score is a way to balance precision and recall to have a single number that indicates how well a model performs. It's particularly useful when the dataset is imbalanced and one class is much more frequent than the other which is the case of my project 
The F1-score is 0.9662894

I must note that these metrics were calculated manually to provide familiarity to the readers with no technical background but can be easily calculated using sklearn.metrics


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





