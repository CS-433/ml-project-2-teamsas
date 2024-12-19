from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVR
import lightgbm as lgb
from src.metrics import measure_performance, get_threshold_for_target


def training_SVR(X, y, test_size, param_grid, target):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    svr = SVR()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(mean_absolute_error, greater_is_better = False)
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        )
    grid_search.fit(X_train, y_train[target])
    best_index = grid_search.best_index_
    best_mean_score = grid_search.cv_results_['mean_test_score'][best_index]
    best_std_score = grid_search.cv_results_['std_test_score'][best_index]
    best_mae = -best_mean_score
    print("Best Parameters:", grid_search.best_params_)
    print("Best MAE score:", best_mae)
    print("Standard Deviation of MAE:", best_std_score)
    y_pred = grid_search.predict(X_test)
    #rmse = root_mean_squared_error(y_test[target], y_pred)
    #mae = mean_absolute_error(y_test[target], y_pred)
    #print("RMSE:", rmse)
    #print("MAE:", mae)
    results = measure_performance(y_true = y_test[target].to_numpy(), y_pred = y_pred, use_classification_metrics = True, thresholds_for_classification = get_threshold_for_target(target))
    print(results)

def Regression_SVR(X, y, type, target, features):

    if type == 'my_personality' and features == 'psycological':
        print(f"results for SVR method on my_personality dataset with {target} target and psycological features:")
        param_grid = {
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.000001, 0.00001, 0.0001, 0.001],
        'kernel': ['poly', 'rbf']
    }
        training_SVR(X, y, 0.2, param_grid, target)    

    if type == 'my_personality' and features == 'embeddings':
        print(f"results for SVR method on my personality dataset with {target} target and Embeddings as features:")
        param_grid = {
        'C': [0.001, 0.01],
        'epsilon': [0.0001, 0.001],
        'kernel': ['rbf']
    }
        training_SVR(X, y, 0.2, param_grid, target)   
    
    if type == 'my_personality' and features == 'combined':
        print(f"results for SVR method on my personality dataset with {target} target and both psycological features and embeddings as features:")
        param_grid = {
        'C': [0.001, 0.01],
        'epsilon': [0.0001, 0.001],
        'kernel': ['rbf']
    }
        training_SVR(X, y, 0.2, param_grid, target)   
        
    if type == 'idiap' and features == 'psycological':
        print(f"results for SVR method on idiap dataset with {target} target and psycological features:")   
        param_grid = {
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.000001, 0.00001, 0.0001, 0.001],
        'kernel': ['poly', 'rbf']
    }
        training_SVR(X, y, 0.1, param_grid, target)   
    
       
    if type == 'idiap' and features == 'embeddings':
        print(f"results for SVR method on idiap dataset with {target} target and Embeddings as features:")    
        param_grid = {
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.000001, 0.00001, 0.0001, 0.001],
        'kernel': ['poly', 'rbf']
    }
        training_SVR(X, y, 0.1, param_grid, target)   
    
    if type == 'idiap' and features == 'combined':
        print(f"results for SVR method on idiap dataset with {target} target and both psycological features and embeddings as features:")
        param_grid = {
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.000001, 0.00001, 0.0001, 0.001],
        'kernel': ['poly', 'rbf']
    }
        training_SVR(X, y, 0.1, param_grid, target)   
    
    if type == 'idiap_chunked' and features == 'psycological':
        print(f"results for SVR method on idiap chunked dataset with {target} target and psycological features:")
        param_grid = {
        'C': [0.001, 0.01, 0.1],
        'epsilon': [0.000001, 0.00001, 0.0001, 0.001],
        'kernel': ['poly', 'rbf']
    }
        training_SVR(X, y, 0.1, param_grid, target)   
    
    if type == 'idiap_chunked' and features == 'embeddings':
        print(f"results for SVR method on idiap chunked dataset with {target} target and Embedding features:")
        param_grid = {
        'C': [0.001, 0.01],
        'epsilon': [0.0001, 0.001],
        'kernel': ['rbf']
    }
        training_SVR(X, y, 0.1, param_grid, target)   
    
    if type == 'idiap_chunked' and features == 'combined':
        print(f"results for SVR method on idiap chunked dataset with {target} target and both psycological features and embeddings as features:")
        param_grid = {
        'C': [0.001, 0.01],
        'epsilon': [0.0001, 0.001],
        'kernel': ['rbf']
    }
        training_SVR(X, y, 0.1, param_grid, target)   
    
def training_LGB(X, y, test_size, param_grid, target):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    model = lgb.LGBMRegressor(random_state=42, verbose = -1, metric='mae' , bagging_fraction=0.8, bagging_freq=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=mae_scorer,
        cv=cv,
        )
    grid_search.fit(X_train, y_train[target])
    best_index = grid_search.best_index_
    best_mean_score = grid_search.cv_results_['mean_test_score'][best_index]
    best_std_score = grid_search.cv_results_['std_test_score'][best_index]
    best_mae = -best_mean_score
    print("Best Parameters:", grid_search.best_params_)
    print("Best MAE score:", best_mae)
    print("Standard Deviation of MAE:", best_std_score)
    y_pred = grid_search.predict(X_test)
    #rmse = root_mean_squared_error(y_test[target], y_pred)
    #mae = mean_absolute_error(y_test[target], y_pred)
    #print("RMSE:", rmse)
    #print("MAE:", mae)
    results = measure_performance(y_true = y_test[target].to_numpy(), y_pred = y_pred, use_classification_metrics = True, thresholds_for_classification = get_threshold_for_target(target))
    print(results)
    

def Regression_LGB(X, y, type, target, features):
    if type == 'my_personality' and features == 'psycological':
        print(f"results for LGB method on my_personality dataset with {target} target and psycological features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70, 80, 90]
        }
        training_LGB(X, y, 0.2, param_grid, target)

    if type == 'my_personality' and features == 'embeddings':
        print(f"results for LGB method on my_personality dataset with {target} target and embedding features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70]
        }
        training_LGB(X, y, 0.2, param_grid, target)

    if type == 'my_personality' and features == 'combined':
        print(f"results for LGB method on my_personality dataset with {target} target and psycological features and embeddings features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70]
        }
        training_LGB(X, y, 0.2, param_grid, target)
        
    if type == 'idiap' and features == 'psycological':
        print(f"results for LGB method on idiap dataset with {target} target and psycological features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70, 80, 90]
        }
        training_LGB(X, y, 0.1, param_grid, target)
        
    if type == 'idiap' and features == 'embeddings':
        print(f"results for LGB method on idiap dataset with {target} target and embedding features:")   
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70]
        }
        training_LGB(X, y, 0.1, param_grid, target)
        
    if type == 'idiap' and features == 'combined':
        print(f"results for LGB method on idiap dataset with {target} target and psycological features and embeddings features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70]
        }
        training_LGB(X, y, 0.1, param_grid, target)
        
    if type == 'idiap_chunked' and features == 'psycological':
        print(f"results for LGB method on idiap chunked dataset with {target} target and psycological features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70, 80, 90]
        }
        training_LGB(X, y, 0.1, param_grid, target)
        
    if type == 'idiap_chunked' and features == 'embeddings':
        print(f"results for LGB method on idiap chunked dataset with {target} target and embedding features:")   
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70]
        }
        training_LGB(X, y, 0.1, param_grid, target)
        
    if type == 'idiap_chunked' and features == 'combined':
        print(f"results for LGB method on idiap chunked dataset with {target} target and psycological features and embeddings features:")
        param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'lambda_l1' : [0.001, 0.01, 0.1],
        'features_fraction' : [70]
        }
        training_LGB(X, y, 0.1, param_grid, target)
        
             