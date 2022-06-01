import numpy as np
import utils.helper as h

import mlflow
import eli5
from sklearn.model_selection import cross_val_score, cross_validate

def get_or_create_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        mlflow.create_experiment(name)
        return mlflow.get_experiment_by_name(name)
    
    return experiment

def _eid(name):
    return get_or_create_experiment(name).experiment_id


def convert_target(y, method='identity'):
    
    if method=="identity":
        return y
    
    elif method=="log":
        return np.log(y)
    
    elif method=="log1p":
        return np.log1p(y)


def unconvert_target(y, method='identity'):
    
    if method=="identity":
        return y
    
    elif method=="log":
        return np.exp(y)
    
    elif method=="log1p":
        return np.expm1(y)
    

def mlflow_start_run(
    df,
    model,
    feats,
    target, 
    experiment_id="dwsolution_property",
    run_name = None,
    convert_target_method='identity',
):
    
    model_name = str(model).split("(")[0]
    if not run_name: 
        run_name = model_name        
    
    X_train, X_test, y_train = h.get_X_y(df=df, feats=feats, target=target)
    y_train = convert_target(y_train, method=convert_target_method)
    

    with mlflow.start_run(experiment_id=_eid(experiment_id), run_name=run_name) as run:

        mlflow.log_params(model.get_params())
        mlflow.log_param("model", model_name)
        mlflow.log_param("feats", feats)
        mlflow.log_param("target", target)
        mlflow.log_param("convert_target_method", convert_target_method)
        mlflow.log_param("X_train.shape", X_train.shape)

        model.fit(X_train, y_train)

        #artifcats
        result = eli5.show_weights(model, feature_names=list(feats))
        with open("../output/eli5.html", "w") as f:
            f.write("<html>{}</html>".format(result.data))
        mlflow.log_artifact("../output/eli5.html", "plot")

        #metrics
        scoring = [
            "neg_mean_absolute_error", 
            "neg_mean_squared_error",  
            "neg_median_absolute_error", 
            "r2", 
        ]
        result = cross_validate(model, X_train, y_train, scoring=scoring, return_train_score=True, return_estimator=False)
        mlflow.log_metrics(
            { 
               ( "avg_{}".format(k) ) : ( -1*np.mean( unconvert_target(-1*v) ) if  k !='r2' else np.mean(v) ) for k, v in result.items() }
        )
        mlflow.log_metrics(
            {
               ( "std_{}".format(k) ) : ( -1*np.std( unconvert_target(-1*v) ) if  k !='r2' else np.std(v) ) for k, v in result.items()
            }
        )