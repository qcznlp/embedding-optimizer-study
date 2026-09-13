"""Exact pre-update gate; no tolerance and no optimizer or clipping side effect."""

def validate(reference, cold_features, cold_local, warm_features, warm_local,
             warm_post_ddp, rng_after_cold, rng_after_warm, recursive_equal):
    if len(warm_local) != 134 or len(warm_post_ddp) != 134 or any(len(v)!=4 for v in warm_local.values()):
        raise ValueError('Require all 134 parameters and four local contributions')
    if len(cold_features) != 4 or warm_features != cold_features:
        raise ValueError('Different actual cold/warm token features')
    if cold_local != warm_local or warm_local != reference['warm_gradient_event_fingerprints']:
        raise ValueError('Local gradients differ from the fixed paired reference')
    if warm_post_ddp != reference['warm_ddp_gradient_fingerprints']:
        raise ValueError('Post-DDP gradients differ from the fixed warm reference')
    if (not reference['same_weights_optimizer_scheduler'] or not reference['same_first_input_rng']
            or not reference['exact_token_features'] or reference['optimizer_updates'] != 0):
        raise ValueError('Invalid paired reference conditions')
    recursive_equal(rng_after_cold,rng_after_warm,'post-pass-rng')
    return {'all_134_post_ddp_tensors_exact_to_reference':True,
            'all_536_local_contributions_exact_to_reference':True,
            'post_pass_rng_equal':True,'actual_token_features_equal':True,'new_tolerance':False}
