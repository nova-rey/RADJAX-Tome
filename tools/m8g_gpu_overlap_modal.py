from __future__ import annotations
import hashlib, json, os, pathlib, platform, resource, shutil, subprocess, time, uuid
import modal

app = modal.App('m8g-gpu-overlap-verification')
TOME='/home/nyx/m8g/repos/tome/src'
CONTRACT='/home/nyx/m8g/repos/contract/src'
MANIFEST='/home/nyx/m8g/evidence/M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1/derived_dataset_manifest.json'
image=(modal.Image.debian_slim(python_version='3.11').pip_install('numpy','torch','psutil','jsonschema','pydantic','cbor2').add_local_dir(TOME,'/workspace/tome/src').add_local_dir(CONTRACT,'/workspace/contract/src').add_local_file(MANIFEST,'/workspace/k_manifest.json'))
vol=modal.Volume.from_name('m8g-gpu-overlap-verification',create_if_missing=True)

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return 'sha256:'+h.hexdigest()

@app.function(image=image,gpu='T4',timeout=3600,volumes={'/mnt/overlap':vol})
def run_suite():
 import torch, sys
 sys.path[:0]=['/workspace/tome/src','/workspace/contract/src']
 os.environ['PYTHONPATH']='/workspace/tome/src:/workspace/contract/src'
 from radjax_tome.builder.delivery.simple_compact_body import write_compact_body_store_from_compact, write_compact_body_store_pipelined_from_compact
 from radjax_tome.builder.delivery.gpu_overlap import descriptor_stream_from_cuda
 ks=[int(x['logical_k']) for x in json.loads(pathlib.Path('/workspace/k_manifest.json').read_text())['records']]
 vocab=max(ks)
 out=pathlib.Path('/mnt/overlap/runs'); out.mkdir(parents=True,exist_ok=True)
 env={'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),'cpu':platform.processor(),'cpu_count':os.cpu_count(),'platform':platform.platform()}
 def make_payload(k, idx, ids, probs, logs):
  return {'selected_example_id':f'synthetic_{idx:04d}','selected_position':idx,'vocab_size':vocab,'num_buckets':3,'effective_top_k':k,'top_token_ids':ids,'top_probs':probs,'top_log_probs':logs,'top_mass':0.5,'tail_mass':0.5,'bucket_masses':(0.2,0.3,0.5),'linkage':None}
 def gpu_batch(start, shaped):
  payloads=[]
  for idx in range(start,min(start+8,len(ks))):
   k=ks[idx]
   ids=torch.arange(k,device='cuda',dtype=torch.int64).add_(idx*17).remainder(vocab)
   probs=torch.full((k,),0.5/k,device='cuda',dtype=torch.float32)
   logs=torch.log(probs)
   if shaped == "production_shaped":
    x=torch.ones((256,256),device='cuda',dtype=torch.float32)
    for _ in range(3): x=x@x.t()*0.0001
   payloads.append((idx,k,ids,probs,logs))
  return payloads
 def serial(cadence, rid):
  root=out/rid; root.mkdir()
  payloads=[]; t0=time.perf_counter()
  for start in range(0,len(ks),8):
   for idx,k,ids,probs,logs in gpu_batch(start,cadence):
    torch.cuda.synchronize(); payloads.append(make_payload(k,idx,ids.cpu(),probs.cpu(),logs.cpu()))
  prod=time.perf_counter()-t0
  t=time.perf_counter(); res=write_compact_body_store_from_compact(root,payloads); total=time.perf_counter()-t0
  return {'variant':'serial_compact','producer_seconds':prod,'representation_seconds':time.perf_counter()-t,'total_seconds':total,'writer':res,'root':str(root), 'body_digest_root':hashlib.sha256(json.dumps(res['body_digests'],separators=(',',':')).encode()).hexdigest(), 'metadata_sha256':res['metadata_sha256']}
 def pipeline(cadence, rid):
  root=out/rid
  stream=descriptor_stream_from_cuda(ks,vocab_size=vocab,batch_size=8,cadence=cadence,torch_module=torch)
  t=time.perf_counter(); res=write_compact_body_store_pipelined_from_compact(root,stream,worker_count=2); total=time.perf_counter()-t
  return {'variant':'pipelined_compact','total_seconds':total,'writer':res,'root':str(root), 'body_digest_root':hashlib.sha256(json.dumps(res['body_digests'],separators=(',',':')).encode()).hexdigest(), 'metadata_sha256':res['metadata_sha256'], **{k:v for k,v in res.items() if k not in {'body_digests','metadata_path','metadata_sha256'}}}
 def lower(cadence,rid):
  t=time.perf_counter();
  for start in range(0,len(ks),8): gpu_batch(start,cadence)
  torch.cuda.synchronize(); return {'variant':'producer_lower_bound','total_seconds':time.perf_counter()-t,'root':None}
 results=[]
 for cadence in ('maximum_pressure','production_shaped'):
  for rep in range(3):
   for variant in ('producer_lower_bound','serial_compact','pipelined_compact'):
    rid=f'{cadence}-{rep}-{variant}-{uuid.uuid4().hex[:8]}'
    rec=(lower(cadence,rid) if variant=='producer_lower_bound' else serial(cadence,rid) if variant=='serial_compact' else pipeline(cadence,rid))
    rec.update({'cadence':cadence,'rep':rep,'run_id':rid})
    if rec.get('root'):
     root=pathlib.Path(rec['root']); rec['output_bytes']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()); shutil.rmtree(root,ignore_errors=True)
    results.append(rec)
 for cadence in ('maximum_pressure','production_shaped'):
  for rep in range(3):
   group=[x for x in results if x['cadence']==cadence and x['rep']==rep and x['variant']!='producer_lower_bound']
   roots={x.get('body_digest_root') for x in group}; metas={x.get('metadata_sha256') for x in group}
   for x in group: x['logical_equivalence']=(len(roots)==1 and len(metas)==1 and x.get('body_digest_root') in roots)
 pathlib.Path('/mnt/overlap/results.json').write_text(json.dumps({'environment':env,'k_count':len(ks),'full_width_count':sum(k==vocab for k in ks),'results':results},sort_keys=True,indent=2))
 return json.dumps({'environment':env,'result_count':len(results),'results_path':'/mnt/overlap/results.json'})

@app.local_entrypoint()
def main():
 print(run_suite.remote())
