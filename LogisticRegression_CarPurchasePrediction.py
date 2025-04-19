import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("Social_Network_Ads.csv", usecols=['Age', 'EstimatedSalary_K', 'Purchased'])

data.head()

data.describe()

from collections import Counter
print(Counter(data['Purchased']))

X = data[['Age', 'EstimatedSalary_K']]
y = data['Purchased']

#Chia du lieu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Chuan bi model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.intercept_)
print(model.coef_)

#Danh gia mo hinh
#  Đánh giá score 
print('Train accuracy score: ', model.score(X_train,y_train))
print('Test accuracy  score: ', model.score(X_test,y_test))

#  Sử dụng các đại lượng khác đánh giá tập test 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred,average='macro')
recall = recall_score(y_test,y_pred,average='macro')
f1 = f1_score(y_test,y_pred)
print(score, recall, precision, f1)

# Visualize_data(X_test.Age,X_test.EstimatedSalary_K,Y_test,'Data test')
plt.figure(figsize=(6,5))
sns.scatterplot(x=X_test.Age, y=X_test.EstimatedSalary_K, hue=y_test, palette="viridis")

x0=np.min(X_test.Age)
x1=np.max(X_test.Age)
y0=-(model.intercept_ + x0*model.coef_[0][0])/model.coef_[0][1]
y1=-(model.intercept_ + x1*model.coef_[0][0])/model.coef_[0][1]
plt.plot([x0,x1],[y0,y1],'r')

plt.title('Data test')
plt.show()

from sklearn.metrics import classification_report,confusion_matrix
#   Vẽ confusion Matrix và nhận xét trên tập test
cm = confusion_matrix(y_test, model.predict(X_test))
cm_df = pd.DataFrame(cm)
plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True,fmt='g')
plt.title('Accuracy:{0:.3f}'.format(score))
plt.ylabel('True Values')
plt.xlabel('Predicted Values')
plt.show()

#  Vẽ ROC_AUC và nhận xét kết quả trên tập test
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
y_prob = model.predict_proba(X_test)[:,1]

# calculate roc curves
fpr, tpr, threshold = roc_curve(y_test, y_prob)

# calculate scores
model_auc = roc_auc_score(y_test, y_prob)
# plot the roc curve for the model
plt.plot([0,1], [0,1], linestyle='--', label='No Skill' )
plt.plot(fpr, tpr, marker='.', label='Model - AUC=%.3f' %(model_auc))
# show axis labels and the legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

# calculate roc curves
y_prob = model.predict_proba(X_test)[:,1]
precision, recall, threshold = precision_recall_curve(y_test, y_prob)
# calculate scores
model_auc = auc(recall, precision)
# plot the roc curve for the model
ns=len(y[y==1]) / len(y)
plt.plot([0,1], [ns,ns], linestyle='--', label='No Skill' )
plt.plot(recall, precision, marker='.', label='Model - AUC=%.3f' %  (model_auc))

# show axis labels and the legend
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Hieu ve chinh nguong
y_proba = model.predict_proba(X_test)[:,1]
y_proba = model.predict_proba(X_test)[:,1]
y_proba

y_predict = y_proba >= 0.45
y_predict

# calculate roc curves
precision, recall, threshold = precision_recall_curve(y_test, y_prob)

scores= (2 * precision * recall)/(precision + recall) 
pos= np.argmax(scores)
print(threshold[pos],scores[pos])

scores = []
thresholds = []
y_proba = model.predict_proba(X_test)[:,1]
for threshold in np.arange(0,1,0.05):
    y_pred = (y_proba >= threshold)
    scores.append(f1_score(y_test, y_pred))
    thresholds.append(threshold)
print(scores)
