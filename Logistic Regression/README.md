#### Logistic Regression : 
- Logistic Regression is a supervised machine learning algorithm used for classification problems.
- it predicts the probability that an input belongs to a specific class.
- It is used for binary classification where the output can be one of two possible categories such as Yes/No, True/False or 0/1.
- It uses sigmoid function to convert inputs into a probability value between 0 and 1.

##### Tupes of Logistic Regression - 
1. Binomial Logistic Regression:
   - This type is used when the dependent variable has only two possible categories. Examples include Yes/No, Pass/Fail or 0/1.
   - It is the most common form of logistic regression and is used for binary classification problems.
    
2. Multinomial Logistic Regression:
   - This is used when the dependent variable has three or more possible categories that are not ordered. For example,
   - classifying animals into categories like "cat," "dog" or "sheep". It extends the binary logistic regression to handle           multiple classes.
     
3. Ordinal Logistic Regression:
   - This type applies when the dependent variable has three or more categories with a natural order or ranking.
   - Examples include ratings like "low," "medium" and "high." It takes the order of the categories into account when modeling.

#### The entire process is summarized in the following algorithm -
initialize: a1=0, a2=0, ..., ak=0, b=0 
- n = 1000 (number of iterations)      
- for (i = 1 to n) 
- { 
  - compute gradients: 
  - da1, da2, ..., da_k, db 
  - update parameters:  
  - a1 = a1 - alpha * da1 
  - a2 = a2 - alpha * da2
  - .
  - . 
  - . 
  - ak = ak - alpha * da_k
  - b = b - alpha * db  
- } 
                
                
