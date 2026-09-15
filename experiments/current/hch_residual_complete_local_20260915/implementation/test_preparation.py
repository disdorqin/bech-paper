import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT/'src')]
from mvp.hch_residual_complete_local_v4_1 import residual_complete_targets,post_level_history,assert_rcl_contract
def main():
 r=np.array([[3.,1.],[2.,6.]],dtype=np.float32);b=np.array([1.,2.],dtype=np.float32);t=residual_complete_targets(r,b)
 assert np.allclose(t['q'],t['delta'][:,None]+t['rho']) and np.allclose(t['rho'].sum(1),0)
 # Keys now follow the core contract (ResidualCompleteTargets), because
 # `core.targets` is the single math authority: amplitude / shape_positive /
 # shape_negative replace the preparation-only A / S_plus / S_minus names.
 assert np.allclose(t['q'],t['delta'][:,None]+t['amplitude'][:,None]*(t['shape_positive']-t['shape_negative']))
 h=post_level_history(r[None],b[None]);assert np.allclose(h['q'],r[None]-b[None,:,None])
 ids=np.array([10,11]);assert_rcl_contract(train_ids=np.array([1,10,11]),local_train_ids=ids,oof_level_ids=ids,level_seen_by_local=ids,level_target_definition='frozen_stage1_level',history_space='post_level_q',normalization_space='local_train_q_geometry')
 for kw in ({'history_space':'raw_residual'},{'normalization_space':'v39_scale'}):
  x=dict(train_ids=np.array([1,10,11]),local_train_ids=ids,oof_level_ids=ids,level_seen_by_local=ids,level_target_definition='frozen_stage1_level',history_space='post_level_q',normalization_space='local_train_q_geometry');x.update(kw)
  try:assert_rcl_contract(**x);raise AssertionError('guard did not fail')
  except ValueError:pass
 print('RCL_PREPARATION_TESTS_PASS')
if __name__=='__main__':main()
