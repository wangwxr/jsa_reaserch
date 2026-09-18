#!/usr/bin/env python3
"""Launch verified Mean-CRAM training code with isolated Hard-CRAM imports."""
from __future__ import annotations
import argparse,json,os,runpy,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--seed',type=int,default=12345);p.add_argument('--gpu',type=int,required=True);p.add_argument('--cram-negative-aggregation',choices=('mean','softmin'),required=True);p.add_argument('--cram-softmin-tau',type=float);p.add_argument('--run-tag');p.add_argument('--resume',action='store_true');args=p.parse_args()
 if args.seed!=12345:raise ValueError('only preregistered seed 12345')
 if args.cram_negative_aggregation=='softmin' and not(args.cram_softmin_tau and args.cram_softmin_tau>0):raise ValueError('positive tau required')
 if args.cram_negative_aggregation=='mean' and args.cram_softmin_tau is not None:raise ValueError('mean has no tau')
 if args.run_tag:
  if any(x in args.run_tag for x in ('/','\\')):raise ValueError('run tag must be a basename')
  os.environ['HARD_CRAM_RUN_ROOT']=str(HERE/'tau_variants'/args.run_tag)
 cfg=HERE/'configs';cfg.mkdir(exist_ok=True)
 for suffix in ('.npy','.json'):
  src=SOURCE/'configs'/f'{args.dataset}_seed{args.seed}_negative_offsets{suffix}';dst=cfg/src.name
  if not dst.exists():dst.symlink_to(src)
 os.environ['CRAM_NEGATIVE_AGGREGATION']=args.cram_negative_aggregation
 if args.cram_softmin_tau:os.environ['CRAM_SOFTMIN_TAU']=str(args.cram_softmin_tau)
 run=Path(os.environ.get('HARD_CRAM_RUN_ROOT',str(HERE)))/'stage1'/args.dataset/f'seed{args.seed}';run.mkdir(parents=True,exist_ok=True)
 (run/'hard_cram_config.json').write_text(json.dumps(vars(args)|{'source_train':str(SOURCE/'train.py'),'lambda_cram':1.0},indent=2)+'\n')
 sys.path.insert(0,str(HERE));sys.argv=[str(SOURCE/'train.py'),'--dataset',args.dataset,'--group','C3','--seed',str(args.seed),'--gpu',str(args.gpu)]+(['--resume'] if args.resume else [])
 runpy.run_path(str(SOURCE/'train.py'),run_name='__main__')
if __name__=='__main__':main()
