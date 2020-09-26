# E2E SVM model on CKD disease
End to end data analysis, eda, prepocessing and SVM model building on a data set called cronic kidney disease from kaggle.com. The data was taken over a 2-month period in India with 25 features ( eg, red blood cell count, white blood cell count, etc). The target is the 'classification', which is either 'ckd' or 'notckd' - ckd=chronic kidney disease. There are 400 rows.

# Importing Libraries
# Linear Algebra
import numpy as np

# Data Processing
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# Algorithm 

from sklearn.svm import SVC, LinearSVC

# getting the data
kidney_data = pd.read_csv('E:\Data Tales Dataset\kidney_disease.csv')
# Understanding the Attributes
age - age of patient
bp - blood pressure level of patient
sg - Specific gravity is the ratio of the density of the substance to the density of a reference substance.
al - Albumin ( It is a type of protein the liver makes. It's one of the most abundant proteins in the blood. We need a proper balance of albumin to keep fluid from leaking out of blood vessels.)
su - Sugar level in the blood
rbc - It refers to Red blood cells in the blood.
pc - Pus is the result of the body's natural immune system automatically responding to an infection, usually caused by bacteria or fungi.
pcc - Pyuria is a condition that occurs when excess white blood cells, or pus, are present in the urine.Parasites, kidney stones, tumors and cysts, and interstitial cystitis can also lead to pyuria.
ba - Bacteria
bgr - The reference values for a "normal" random glucose test in an average adult are 79–140mg/dl (4.4–7.8 mmol/l), between 140-200mg/dl (7.8–11.1 mmol/l) is considered pre-diabetes, and ≥ 200 mg/dl is considered diabetes according to ADA guidelines
bu - Nitrogen in the blood that comes from urea (a substance formed by the breakdown of protein in the liver). The kidneys filter urea out of the blood and into the urine. A high level of urea nitrogen in the blood may be a sign of a kidney problem.
sc - Serum Creatinine ( Creatinine is a breakdown product of creatinine phosphate in muscle, and is usually produced ata)
sod - Sodium (sod in mEq/L)
pot - Potassium (pot in mEq/L)
hemo - Hemoglobin (hemo in gms)
pcv - Packed Cell Volume
wc - White Blood Cell Count (wc in cells/cumm)
rc - Red Blood Cell Count(rc in millions/cumm)
htn - Hypertension (It is also known as high blood pressure(HBP) is a long-term medical condition in which the blood pressure in the arteries is persistently elevated.)
dm - Diabetes Mellitus(A disease in which the body’s ability to produce or respond to the hormone insulin is impaired, resulting in abnormal metabolism of carbohydrates and elevated levels of glucose in the blood.)
cad - Coronary Artery Disease (It happens when the arteries that supply blood to heart muscle become hardened and narrowed.)
appet - Appetite (A natural desire to satisfy a bodily need, especially for food)
pe - Pedal Edema( It is the accumulation of fluid in the feet and lower legs. )
ane - Anemia (A condition in which there is a deficiency of red cells or of haemoglobin in the blood, resulting in pallor and weariness.)
classification- It classifies whether a person is suffering from chronic kidney disease or not.

# Data decription
kidney_data.describe()

# Visualization of Missing variables
plt.figure(figsize=(20,10))
sns.heatmap(kidney_data.isnull(), yticklabels=False, cbar=False, cmap='plasma')

From graph, we see null values in the columns age, bp, sg, al, su, rbc, pc, pcc,ba,bgr,bu, sc,sod, pot, hemo, pcv, wc, rc, htn, dm ,cad, appet, pe and ane. Let us also find the count of missing values.
# Count of null values
kidney_data.isnull().sum()

# Let us have a look into the categorical attributes which have missing values in them
or cols in kidney_data.select_dtypes("object"):
    kidney_data[cols] = kidney_data[cols].astype("category")
# For few categorical columns we see unidentified data entry error. Hence, we need to replace this with space.
 for cols in kidney_data[de_cols]:
    kidney_data[cols] = kidney_data[cols].str.replace('\t',"")
    kidney_data[cols] = kidney_data[cols].replace("?",np.nan)
    kidney_data[cols] = kidney_data[cols].str.strip()
 # Outlier analysis
 sns.set(font_scale=1.5)
fig = plt.figure(figsize=(24,40))
i=1
for column in kidney_data[num_cols]:
    plt.subplot(5,3,i)
    sns.boxplot(x=kidney_data.classification, y=kidney_data.loc[:,column])
    i = i + 1
    
plt.tight_layout()
plt.show()

Inferences:-
We see some even teenage samples in our data, which is not belonging to the normal range. Blood Pressure so some patients being very high , there can be chances of patients having very high blood pressure. Also, we observe some patients having high sugar levels above normal range.Patients having bgr > 200 are suffering from diabetes according to ADA guidelines. Patients have higher nitrogren levels contained in urea in body. If the Serum Certainine level is very high the person suffers from chronic kidney disease and has very less chance to survive.The normal level to have sodium in body is 135-145 mEq/L,but for kidney failure patients usually have low sodium.But level almost 0 is again an outlier. Potassium levels in body which are greater than 7 , patients have chances of severe hyperkalemia. However, levels greater than 30 is really an outlier. Hemoglobin in a person less than 5 is clearly is outlier and person has very less chance to survive.

# Univariate analysis
# Pie chart showing the data if its checked or not checked ( in our data set)
labels = 'ckd','notckd'
sizes = [kidney_data.classification[kidney_data['classification']=='ckd'].count(), kidney_data.classification[kidney_data['classification']=='notckd']
         .count()]
explode = (0, 0.1)
fig, ax = plt.subplots(figsize=(10,8))
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90, 
       textprops={'fontsize':14})
ax.axis('equal')
plt.legend(loc='upper right')
plt.title("Proportion of patients suffering from chronic-kidney disease",size=20)
plt.show()
# Analysis on cat features
sns.set(rc={'figure.figsize':(24,20)}, font_scale=1.5)

i = 1
for column in kidney_data.select_dtypes("category"):
    if column != "classification":
        plt.subplot(4,3,i)
        sns.countplot(kidney_data[column])
        i = i + 1
                   
plt.tight_layout()
plt.show()
The univariate analysis of various categorical variables shows the distribution of labels in the respective columns. 
# Analysis on numerical variables
sns.set(rc={'figure.figsize':(24,24)}, font_scale=1.5)
i = 1
for column in kidney_data.select_dtypes(["int64","float64"]):
    plt.subplot(5,3,i)
    sns.distplot(kidney_data[column])
    i = i + 1
          
plt.tight_layout()
plt.show()

From this we can check if the data is skew biased , to howmuch extent the distribution is normalised 
 # Bivariate Analysis
 sns.set(style="white")

# Compute the correlation matrix
corr = kidney_data.iloc[:, :-1].corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot= True);
plt.title("CROSS CORRELATION BETWEEN PREDICTORS", fontsize=15)
plt.show()
# Contiunuous features comparison w.r.t target
sns.set()
fig = plt.figure(figsize=(19,20))
i=1
for column in kidney_data.select_dtypes(["int64","float64"]):
    plt.subplot(7,2,i)
    sns.distplot(kidney_data.loc[kidney_data.classification=='ckd',column],hist=False,kde=True,
                kde_kws={'shade': True, 'linewidth':3},
                label='ckd')
    sns.distplot(kidney_data.loc[kidney_data.classification=='notckd',column],hist=False,kde=True,
                kde_kws={'shade': True, 'linewidth':3},
                label='notckd')
    i=i+1
plt.tight_layout()
plt.show()

Inferences from distribution of continuous features
While looking at columns sg, hemo, bu and bgr we see lot of patients of non-chronic kidney disease lying in the high value range.
Patients suffering from chronic ailment don't fall in the normal category in column "age".
In rest other columns, we see lot of bimodal/trimodal distribution present such as "sg","bp" and "pot".

# Comparison of Categorical Features w.r.t Target
def bivariate_cat(data,col1,col2,rot):
    if col2=='classification':
        cross_tab = pd.crosstab(data[col1], data[col2]).apply(lambda x: x/x.sum() * 100, axis=1).round(2)
        cross_tab.plot.bar(figsize=(6,5))
        plt.xlabel('{}'.format(col1))
        plt.ylabel('% of patients who are suffering from chronic-disease'.format(col1))
        plt.title('{} Vs chronic-disease-suffering'.format(col1))
        plt.xticks(rotation=rot)
        plt.show()
        return cross_tab
        
  # using this function to comapare each and every paramter wrt target
   bivariate_cat(kidney_data,'rbc','classification',45)
   we can change the 'rbc; with corresponding columns and get those analysis.
   
# Encoding Categorical Features
   
  from sklearn.preprocessing import LabelEncoder

# target column
tgt_col = ['classification']

# Categorical cols
category_names = kidney_data.nunique()[kidney_data.nunique() < 20].keys().tolist()
category_names = [x for x in category_names if x not in tgt_col]

# Numerical cols
num_cols = [i for i in kidney_data.columns if i not in category_names + tgt_col]

# Binary cols
bin_cols = kidney_data.nunique()[kidney_data.nunique()==2].keys().tolist()

# Multi-cols
multi_cols = [i for i in category_names if i not in bin_cols]

# Label Encoding Binary cols
le = LabelEncoder()
for i in bin_cols:
    kidney_data[i] = le.fit_transform(kidney_data[i])
    
# Duplicating cols for multi-value columns
kidney_data = pd.get_dummies(data=kidney_data, columns=multi_cols)
# Normalising features
cont_features = []
for features in kidney_data.select_dtypes(include=['int64','float64']):
    cont_features.append(features)
kd_df = kidney_data
# Scaling numerical features
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
kd_df[cont_features] = minmax.fit_transform(kd_df[cont_features].values)
# Model Building
cols = [i for i in kd_df.columns if i not in tgt_col]
X = kd_df[cols]
Y = pd.DataFrame(kd_df['classification'])
# Using K fold cross validation
from sklearn.model_selection import KFold
folds = KFold(n_splits=5, shuffle=True, random_state=0)
for train_index, test_index in folds.split(X,Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

# Using SVM Algorithm
I am  using different SVM kernels(linear,gaussian, polynomial) and also tune the various parameters such as C, gamma and degree to find out the best performing model.

# Running SVM with default parameter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
svc = SVC()
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print("Accuracy Score : ")
print(accuracy_score(Y_test, y_pred))
 On my dataset i achived an Accuracy score of 0.915

# Confusion Matrix
confusion_matrix(Y_test, y_pred)
# Classification Report
print(classification_report(Y_test, y_pred))
# Liner Kernal
svc = SVC(kernel='linear')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print("Accuracy Score :")
 On my dataset i achived an Accuracy score of 0.975
 # RBF Kernel
 svc = SVC(kernel='rbf')
svc.fit(X_train, Y_train)
y_pred = svc.predict(X_test)
print("Accuracy Score")
print(accuracy_score(Y_test, y_pred))
On my dataset i achived an Accuracy score of 0.915

# Hyper-parameter Tuning
Large values of C causes "overfitting" of model and small values of "C" causes "underfitting" of model. Thus, the value of C needs to be generalized.
from sklearn.model_selection import cross_val_score
C_range = list(np.arange(0.1,2,0.1))
acc_score = []
for c in C_range:
    svc = SVC(kernel='linear',C=c)
    scores = cross_val_score(svc, X, Y,scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)

# Plotting graph
import matplotlib.pyplot as plt
%matplotlib inline

C_range = list(np.arange(0.1,2,0.1))
# plot the value of C for SVM
plt.plot(C_range, acc_score)
#plt.xticks(np.arange(0.0001,0.1,0.1))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

The accuracy steadily increated till c=0.27 and then it was flat and had pleatues. To overcome overfitting the best c value coyble be 0.27 with accuracy as 0.98



 
 
print(accuracy_score(Y_test, y_pred))
        
