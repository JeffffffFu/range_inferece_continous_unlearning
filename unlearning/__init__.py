from continuous.unlearn.GA import continuous_unlearn_GA
from continuous.unlearn.NPO import continuous_unlearn_NPO
from continuous.unlearn.finetune import continuous_update_finetune
from continuous.unlearn.retrain import continuous_unlearn_retrain
from continuous.unlearn.sparsity import continuous_unlearn_sparsity
from unlearning.GA import GA
from unlearning.IF import IF
from unlearning.NPO import NPO
from unlearning.certified import certified
from unlearning.fisher import fisher
from unlearning.negGrad import NegGrad
from unlearning.retrain import retrain
from unlearning.retrain_dp import retrain_dp
from unlearning.scrub import scrub
from unlearning.sisa import sisa
from unlearning.sparsity import sparsity


def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""
    if name == "retrain":
        return retrain
    elif name == "continuous_unlearn_retrain":
        return continuous_unlearn_retrain
    elif name == "continuous_unlearn_NPO":
        return continuous_unlearn_NPO
    elif name == "continuous_unlearn_GA":
        return continuous_unlearn_GA
    elif name == "continuous_unlearn_sparsity":
        return continuous_unlearn_sparsity
    elif name == "continuous_update_finetune":
        return continuous_update_finetune
    elif name == "GA":
        return GA
    elif name == "sparsity":
        return sparsity
    elif name =='IF':
        return IF
    elif name =='fisher':
        return fisher
    elif name =='scrub':
        return scrub
    elif name =='sisa':
        return sisa
    elif name == "retrain_dp":
        return retrain_dp
    elif name == "certified":
        return certified
    elif name== 'NegGrad':
        return NegGrad
    elif name == "NPO":
        return NPO
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
