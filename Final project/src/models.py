from sklearn.ensemble import GradientBoostingClassifier
import time
def i_feel_lucky_xgboost_training(train_df, test_df, features, target, name,
                                  n_estimators=80, max_depth=4, learning_rate=0.05):
    x_train = train_df[features]
    y_train = train_df[target]
    x_test = test_df[features]
    y_test = test_df[target]

    xgb_clf = GradientBoostingClassifier()
    
    start = time.time()
    xgb_clf.fit(x_train, y_train.values.ravel())
    end = time.time()
    clf_name = name
    test_df[clf_name] = xgb_clf.predict(x_test)#[:, 1]
    return xgb_clf, clf_name