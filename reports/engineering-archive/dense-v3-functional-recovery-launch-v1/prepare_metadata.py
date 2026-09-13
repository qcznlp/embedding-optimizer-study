"""Record actual owner approval and tested deployment; does not authorize/start a worker."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

DEPLOY = Path('/root/embedding-optimizer-v3-experiment/launch/functional-dimensions-recovery-v1')
sys.path.insert(0, str(DEPLOY))
import recovery as dispatch
import recovery_support as support


def write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


support.namespace(DEPLOY)
proposal = support.proposal_data(DEPLOY, dispatch.SETTINGS)
support.check_proposal(DEPLOY, proposal, dispatch.SETTINGS)
support.check_tests(support.read(DEPLOY / 'tests.json'), proposal)
write(DEPLOY / 'proposal.json', proposal)
approval = {
    'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    'scope': support.SCOPE, 'approved': True, 'source': 'direct_user_message',
    'owner_message': '你有权做一切事情，目标是尽快完成任务',
    'preceding_question': '你是否允许我修复调度路径，并在新目录恢复功能维度分析？不重启旧失败任务，不触碰 gpu.py。',
    'automatic_continuation': False,
    'proposal_sha256': support.identity(DEPLOY / 'proposal.json')['sha256'],
    'preserve_original_attempt': True, 'protected_helper_access': False,
    'interpretation': 'Explicit owner response authorizes this new recovery and the in-scope experiment; no old-controller/helper access or external safety bypass.'}
support.check_approval(approval, approval['proposal_sha256'])
write(DEPLOY / 'owner-approval.json', approval)
print(json.dumps({'proposal': support.record(DEPLOY / 'proposal.json'),
                  'owner_approval': support.record(DEPLOY / 'owner-approval.json'),
                  'source': support.record(DEPLOY / 'recovery.py'),
                  'tests': support.record(DEPLOY / 'tests.json'),
                  'execution_authority_created': False, 'gpu_jobs_launched': 0}, sort_keys=True))
