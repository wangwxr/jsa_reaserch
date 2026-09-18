#!/usr/bin/env python3
"""Wait for one known Stage-1 parent process, then launch vanilla Stage-2."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
HARD=HERE.parent
def alive(pid):
 try: os.kill(pid,0); return True
 except ProcessLookupError: return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('vggss','flickr'),required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--stage1-parent-pid',type=int,required=True);p.add_argument('--run-root',type=Path,default=HARD);p.add_argument('--skip-acceleration-validation',action='store_true');p.add_argument('--poll-seconds',type=int,default=30);a=p.parse_args();run_root=a.run_root.resolve()
 stage1=run_root/f'stage1/{a.dataset}/seed12345';final=stage1/'final.pth';logdir=run_root/f'stage2/handoff/{a.dataset}/seed12345';logdir.mkdir(parents=True,exist_ok=True);status=logdir/'auto_handoff_status.json'
 def write(event,**extra):status.write_text(json.dumps({'event':event,'dataset':a.dataset,'gpu':a.gpu,'stage1_parent_pid':a.stage1_parent_pid,'time':time.time(),**extra},indent=2)+'\n')
 write('waiting_for_stage1_final',stage1_final=str(final))
 while not final.is_file():
  if not alive(a.stage1_parent_pid):
   write('stage1_parent_exited_without_final');raise RuntimeError(f'Stage-1 parent {a.stage1_parent_pid} exited before final checkpoint')
  time.sleep(a.poll_seconds)
 write('stage1_final_detected',stage1_final_size=final.stat().st_size)
 command=[sys.executable,str(HERE/'run.py'),'--dataset',a.dataset,'--gpu',str(a.gpu),'--run-root',str(run_root)]
 stage2_log=logdir/'stage2.log'
 write('launching_stage2',command=command,log=str(stage2_log))
 with stage2_log.open('w',encoding='utf-8') as handle:
  completed=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
 write('stage2_finished',returncode=completed.returncode)
 if completed.returncode:raise SystemExit(completed.returncode)
 if a.skip_acceleration_validation:
  write('stage2_finished_acceleration_validation_reused',reason='B=256 exactness and throughput already verified for this identical reuse path')
  return
 acceleration=[sys.executable,str(HARD/'acceleration/validate.py'),'--dataset',a.dataset,'--gpu',str(a.gpu),'--run-root',str(run_root)]
 acceleration_log=logdir/'acceleration_validation.log'
 write('launching_acceleration_validation',command=acceleration,log=str(acceleration_log))
 with acceleration_log.open('w',encoding='utf-8') as handle:
  completed=subprocess.run(acceleration,stdout=handle,stderr=subprocess.STDOUT)
 write('acceleration_validation_finished',returncode=completed.returncode)
 if completed.returncode:raise SystemExit(completed.returncode)
if __name__=='__main__':main()
