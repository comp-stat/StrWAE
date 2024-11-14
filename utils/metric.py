import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

def adult_metric(Z_train, Z_test, S_train, S_test, Y_train, Y_test, cls_name='rf'):
    """
        Original implementation: https://github.com/SoftWiser-group/FairDisCo/blob/main/utils.py
    """
    assert cls_name in {'rf', 'lr'}
    clf = RandomForestClassifier(random_state=1) if cls_name == 'rf' else LogisticRegression(penalty=None, max_iter=5000)

    # sauc
    clf = clf.fit(Z_train, S_train)
    S_test_hat = clf.predict_proba(Z_test)[:,1]
    sauc = roc_auc_score(S_test, S_test_hat)
    if sauc < 0.5: sauc = roc_auc_score(S_test, 1-S_test_hat)
    sacc = clf.score(Z_test, S_test)

    clf2 = RandomForestClassifier(random_state=1) if cls_name == 'rf' else LogisticRegression(penalty=None, max_iter=5000)
    # yauc
    clf2 = clf2.fit(Z_train, Y_train)
    Y_test_hat = clf2.predict_proba(Z_test)[:,1]
    yauc = roc_auc_score(Y_test, Y_test_hat)
    if yauc < 0.5: yauc = roc_auc_score(Y_test, 1-Y_test_hat)
    yacc = clf2.score(Z_test, Y_test)

    # dp
    Y_test_hat = clf2.predict(Z_test)
    Y_test_hat_0_mean = Y_test_hat[S_test==0].mean()
    Y_test_hat_1_mean = Y_test_hat[S_test==1].mean()
    dp = abs(Y_test_hat_0_mean - Y_test_hat_1_mean)

    return {
        "y_auc": yauc,
        "y_acc": yacc,
        "s_auc": sauc,
        "s_acc": sacc,
        "dp": dp
    }


def eyaleb_metric(Z_train, Z_test, S_train, S_test, S_train_c, S_test_c, Y_train, Y_test):

    # Classifier & Regressor
    log_y = LogisticRegression(penalty=None, max_iter=5000).fit(Z_train, Y_train.flatten())
    reg_z = LinearRegression().fit(Z_train, S_train)
    rf_z = RandomForestRegressor(random_state=1).fit(Z_train, S_train)

    reg_z_c = LogisticRegression(penalty=None, max_iter=5000).fit(Z_train, S_train_c)
    rf_z_c = RandomForestClassifier(random_state=1).fit(Z_train, S_train_c)

    acc_log_y = log_y.score(Z_test, Y_test)
    acc_reg_z_c = reg_z_c.score(Z_test, S_test_c)
    acc_rf_z_c = rf_z_c.score(Z_test, S_test_c)

    mse_reg_z = ((reg_z.predict(Z_test) - S_test)**2).sum(axis=1).mean()
    mse_rf_z = ((rf_z.predict(Z_test) - S_test)**2).sum(axis=1).mean()

    print(
        f"acc_y: {acc_log_y}\nacc_reg_z_c: {acc_reg_z_c}\nacc_rf_z_c: {acc_rf_z_c}\n" + \
        f"mse_reg_z: {mse_reg_z}\nmse_rf_z: {mse_rf_z}"
    )

    return {
        "acc_y": acc_log_y,
        "acc_reg_z_c": acc_reg_z_c,
        "acc_rf_z_c": acc_rf_z_c,
        "mse_reg_z": mse_reg_z,
        "mse_rf_z": mse_rf_z
    }