import copy
import pytest
from gate import validate

def values():
    local = {str(i):[str(j) for j in range(4)] for i in range(134)}
    post = {str(i):str(i) for i in range(134)}
    reference = dict(warm_gradient_event_fingerprints=copy.deepcopy(local),
        warm_ddp_gradient_fingerprints=copy.deepcopy(post),same_weights_optimizer_scheduler=True,
        same_first_input_rng=True,exact_token_features=True,optimizer_updates=0)
    def equal(a,b,label):
        if a!=b:
            raise ValueError(label)
    return [reference,[1,2,3,4],copy.deepcopy(local),[1,2,3,4],local,post,123,123,equal]

def test_exact_gate_accepts_complete_pair():
    result=validate(*values())
    assert result['all_134_post_ddp_tensors_exact_to_reference'] and not result['new_tolerance']

@pytest.mark.parametrize('change',['parameter_count','event_count','features','cold_local','reference_local','post_ddp','rng'])
def test_any_boundary_difference_refuses_before_update(change):
    v=values()
    if change=='parameter_count': v[4].pop('0')
    elif change=='event_count': v[4]['0'].pop()
    elif change=='features': v[3][0]=0
    elif change=='cold_local': v[2]['0'][0]='different'
    elif change=='reference_local': v[0]['warm_gradient_event_fingerprints']['0'][0]='different'
    elif change=='post_ddp': v[5]['0']='different'
    elif change=='rng': v[7]=456
    with pytest.raises(ValueError): validate(*v)

@pytest.mark.parametrize('field',['same_weights_optimizer_scheduler','same_first_input_rng',
                                 'exact_token_features','optimizer_updates'])
def test_invalid_reference_conditions_refuse(field):
    v=values()
    v[0][field]=1 if field=='optimizer_updates' else False
    with pytest.raises(ValueError): validate(*v)
