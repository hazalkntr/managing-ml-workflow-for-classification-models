# The df set used in this example is from http://archive.ics.uci.edu/ml/dfsets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by df mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#for ML
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix, precision_score, recall_score

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file 
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    df = pd.read_csv(wine_path)

    #del df['Id']

    for col in ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']:
        df[col]=df[col]/df[col].max()

    feature=np.array(df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']])
    label=np.array(df['quality'])

    xtrain,xtest,ytrain,ytest=train_test_split(feature,label,test_size=0.2,random_state=0)

    model_comp={}

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        rf=RandomForestClassifier()
        rf.fit(xtrain,ytrain)
        y3=rf.predict(xtest)

        acc=accuracy_score(ytest,y3)
        f1=f1_score(ytest,y3,average='weighted')
        prec=precision_score(ytest,y3,average='weighted')
        recal=recall_score(ytest,y3,average='weighted')

        print(acc)
        print(f1)
        print(prec)
        print(recal)
        print(classification_report(ytest,y3))

        model_comp['Random forest']=[acc,f1,prec,recal]


        # model comparison
        df1=pd.DataFrame.from_dict(model_comp).T
        df1.columns=['Accuracy','F1_score','Precision','Recall']
        df1=df1.sort_values('F1_score',ascending=True)
        df1.style.background_gradient(cmap='Greens')


        mlflow.log_metric("accuracy score", acc)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("precision score", prec)
        mlflow.log_metric("recall score", recal)


        predictions = rf.predict(xtrain)
        signature = infer_signature(xtrain, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


        if tracking_url_type_store != "file":
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                rf, "model", registered_model_name="RandomForestModel", signature=signature
            )
        else:
            mlflow.sklearn.log_model(rf, "model", signature=signature)
        
