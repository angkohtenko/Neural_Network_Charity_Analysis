# Neural Network Charity Analysis
## Overview of the analysis

The goal of the project was to create and train deep learning model on [Alphabet Soup Charity dataset]( https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/f281f5df7678ec25af9efafe7af135a3e2f91f6c/Resources/charity_data.csv) in order to predict whether applicants will be successful if funded by charity organization - Alphabet Soup.

To complete the project I used Python along with pandas, sklearn and tensorflow libraries.

## Results

### Data Preprocessing

Original dataset has following columns:
![dataset](https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/main/images/dataset.png)

-	EIN and Name columns were dropped as they don’t bring any value to the analysis.
-	Column IS_SUCCESSFUL is considered as target.
-	The rest of the columns were used as features for the model.

### Compiling, Training, and Evaluating the Model

I’ve created a model with 2 hidden layers. First and second layers have 80 and 30 neurons respectively. Both layers use Relu as an activation function. Output layer uses Sigmoid as an activation function.

First model predicts target with accuracy 0.731
![perf_model_1](https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/main/images/perf_model_1.png)

To achieve goal of 75% accuracy, at first, I’ve increased neurons for the second layer up to 80.
Accuracy of that model decreased to 0.727
![perf_model_2](https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/main/images/perf_model_2.png)

Then, I tried to change activation function for hidden layers to Tanh.
Accuracy of that model was close to the initial model - 0.73.
![perf_model_3](https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/main/images/perf_model_3.png)

Next, I’ve applied bucketing technique for asked amount (column ASK_AMT). I’ve created bins and cut the dataframe:
```
ask_amount_bins = [0, 5000, 15000, 50000, 200000, 500000, 9000000000]
bin_names = ['<5000', '5000-15000', '15000-50000', '50000-200000', '200000-500000', '>500000']
application_df['ask_amount_bins'] = pd.cut(application_df.ASK_AMT, ask_amount_bins, labels=bin_names)
application_df.drop(columns='ASK_AMT', inplace=True)
```
![df_bins](https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/main/images/df_bins.png)

For the model I’ve used the initial parameters and get the accuracy 0.728
![perf_model_4](https://github.com/angkohtenko/Neural_Network_Charity_Analysis/blob/main/images/perf_model_4.png)
## Summary
I didn’t reach the goal of 75% accuracy for the model. 

As potential improvements I would try to remove such columns as application type and classification. They may occur to be noisy feature.

The increase of quantity of neurons seems to be insufficient. The changes should be done in preprocessing dataset.
