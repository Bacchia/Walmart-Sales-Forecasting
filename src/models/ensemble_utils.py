import numpy as np

def calculate_blended_predictions(xgb_preds, lgb_preds, nn_preds, weights):
    w_xgb = weights.get('xgb_weight', 0.0)
    w_lgb = weights.get('lgb_weight', 0.0)
    w_nn = weights.get('nn_weight', 0.0)
    
    total_w = w_xgb + w_lgb + w_nn
    if not np.isclose(total_w, 1.0):
        w_xgb, w_lgb, w_nn = w_xgb/total_w, w_lgb/total_w, w_nn/total_w

    final_preds = (w_xgb * np.array(xgb_preds)) + \
                  (w_lgb * np.array(lgb_preds)) + \
                  (w_nn * np.array(nn_preds))
    return final_preds
