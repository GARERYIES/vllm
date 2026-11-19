"""SM80 pairing on the pinned upstream source. Kernel function is shared by both wrappers."""
import json
import random
import statistics
from pathlib import Path

import torch
import main_fp32_pair as main_pair
import lab_inputs as lab

assert torch.cuda.device_count()==1 and torch.cuda.get_device_capability()==(8,0)
torch.backends.cuda.matmul.allow_tf32=False
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
out=parser.parse_args().out
out.mkdir(parents=True,exist_ok=False)
rng=random.Random(8311)
base=main_pair.fused_recurrent_gated_delta_rule_packed_decode
cand=main_pair.candidate
assert base.__globals__["fused_recurrent_gated_delta_rule_packed_decode_kernel"] is cand.__globals__["fused_recurrent_gated_delta_rule_packed_decode_kernel"]
def call(fn, inputs, state=None, result=None):
    m,a,b,al,dt,s,i,o=inputs
    return fn(m,a,b,al,dt,128**-.5,s if state is None else state,o if result is None else result,i,True)
def check_shape(b,h,hv,sd,steps):
    sdt=torch.float32 if sd=="fp32" else torch.bfloat16
    initial=lab.make_inputs(b,h,hv,128,128,sdt,seed=200)
    sb=initial[5].clone(); sc=sb.clone(); sr=sb.clone()
    maxima=[0.,0.,0.,0.]
    bitwise=True
    for step in range(steps):
        xs=lab.make_inputs(b,h,hv,128,128,sdt,seed=200+step) if step else initial
        ob=torch.empty_like(xs[-1]); oc=torch.empty_like(ob)
        call(base,xs,sb,ob); call(cand,xs,sc,oc)
        bitwise=bitwise and torch.equal(ob,oc) and torch.equal(sb,sc)
        maxima[0]=max(maxima[0],(ob.float()-oc.float()).abs().max().item())
        # Restrict peak memory in the independent, dense FP32 reference.
        for lo in range(0,b,8):
            hi=min(lo+8,b)
            m,a,be,al,dt,_,ix,_=xs
            ids=ix[lo:hi].long()
            yr,hr=lab.ref_torch(m[lo:hi],a[lo:hi],be[lo:hi],al,dt,128**-.5,sr,ids,h,128)
            sr[ids]=hr.to(sdt)
            hh=sc[ids].float()
            bb=sb[ids].float()
            yy=oc[lo:hi,0].float()
            maxima[1]=max(maxima[1],(hh-bb).abs().max().item())
            maxima[2]=max(maxima[2],(yy-yr).abs().max().item())
            maxima[3]=max(maxima[3],(hh-hr).abs().max().item())
            torch.testing.assert_close(yy,yr,atol=.02,rtol=.01)
            torch.testing.assert_close(hh,hr,atol=.02,rtol=.01)
        assert torch.equal(sb[0],sc[0]) and torch.equal(sb[0],initial[5][0])
        assert maxima[0]<=1e-6 and maxima[1]<=1e-6, (b,h,hv,sd,step,maxima)
    return dict(bitwise=bitwise,abs_max=maxima,steps=steps)
def bench(b,h,hv,sd):
    dt=torch.float32 if sd=="fp32" else torch.bfloat16
    sets=lab.make_input_sets(b,h,hv,128,128,dt)
    arms={"base":lambda *xs:call(base,xs),"cand":lambda *xs:call(cand,xs)}
    # Independently warm both; five counterbalanced pairs on identical input addresses.
    for arm in arms.values():arm(*sets[0])
    torch.cuda.synchronize()
    pairs=[]
    flip=rng.randrange(2)
    for rep in range(5):
        order=("base","cand") if (rep+flip)%2 else ("cand","base")
        us={arm:lab.bench_graph(arms[arm],sets,iters=15) for arm in order}
        pairs.append([*order,us["base"],us["cand"],us["base"]/us["cand"]])
    return pairs
shapes=[]
for sd in ("fp32","bf16"):
    for hv in (8,12,16,24,32,48,64):
        for ratio,batches in ((2,[1,8,16,32,64]),(4,[1,8,16,32,64]),(2,[2,4,24,48,128,256])):
            for b in batches:shapes.append(("matrix",b,hv//ratio,hv,sd))
    for b in (1,4,8,16,32,64):shapes.append(("27B-TP2",b,8,24,sd))
rng.shuffle(shapes)
for i,(group,b,h,hv,sd) in enumerate(shapes):
    try:
        correctness=check_shape(b,h,hv,sd,3)
        torch.cuda.empty_cache()
        pairs=bench(b,h,hv,sd)
        row=dict(group=group,b=b,h=h,hv=hv,sd=sd,correctness=correctness,pairs=pairs,
                 speed=statistics.median(x[-1] for x in pairs))
    except BaseException:
        print("FAILED",group,b,h,hv,sd,flush=True)
        raise
    with (out/"rows.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
    if i%20==0:print("complete",i+1,"/",len(shapes),flush=True)
print("complete",len(shapes),flush=True)
