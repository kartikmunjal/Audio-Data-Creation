import numpy as np
from scripts.run_learned_filter import fit_predict, metrics


def test_locked_classifier_is_deterministic_and_reports_confusion_matrix():
    x=np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]],dtype=float)
    y=np.array([False,False,False,True,True,True])
    first,_=fit_predict(x,y,x); second,_=fit_predict(x,y,x)
    assert np.array_equal(first,second)
    report=metrics(y,first)
    assert report["n"]==6
    assert len(report["confusion_matrix_tn_fp_fn_tp"])==4
