"""Exercise the candidate image's unmodified caption handler on Blackwell."""
import base64,hashlib,json,os,platform,runpy,time,sys
from pathlib import Path
import torch,torchvision,transformers
ROOT=Path('/opt/joycaption-smoke');ROOT.mkdir(exist_ok=True)
assert torch.__version__=='2.8.0+cu128',torch.__version__
assert torchvision.__version__=='0.23.0+cu128',torchvision.__version__
assert torch.version.cuda=='12.8',torch.version.cuda
assert transformers.__version__=='4.51.3',transformers.__version__
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability()==(12,0),torch.cuda.get_device_capability()
assert 'Blackwell' in torch.cuda.get_device_name()
torch.ones((32,32),device='cuda',dtype=torch.bfloat16).matmul(torch.ones((32,32),device='cuda',dtype=torch.bfloat16));torch.cuda.synchronize()
metadata={'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'total_vram_bytes':torch.cuda.get_device_properties(0).total_memory,'model_source':os.environ['MODEL_DIR'],'cuda_tensor_check':'passed'}
(ROOT/'runtime.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)
import runpod
registered=[]
runpod.serverless.start=lambda config:registered.append(config['handler'])
sys.path.insert(0,'/app')
started=time.monotonic();runpy.run_path('/app/handler.py',run_name='joycaption_candidate_smoke');load_s=time.monotonic()-started
assert len(registered)==1
fixture=(ROOT/'fixture.jpg').read_bytes()
results=[]
for index in range(2):
 torch.manual_seed(42)
 started=time.monotonic()
 inputs={'image':'data:image/jpeg;base64,'+base64.b64encode(fixture).decode(),'prompt':'Write a descriptive caption for this image.','max_new_tokens':256,'temperature':0.6,'top_p':0.9}
 result=registered[0]({'id':'joycaption-blackwell-smoke-'+str(index),'input':inputs})
 torch.cuda.synchronize()
 assert not result.get('error'),result
 assert len(result.get('caption','').strip())>30,result
 assert result['model']=='fancyfeast/llama-joycaption-beta-one-hf-llava',result
 row={'index':index,'elapsed_s':time.monotonic()-started,'result':result,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
 results.append(row);print(json.dumps(row),flush=True)
 report={'runtime':metadata,'model_load_s':load_s,'fixture_sha256':hashlib.sha256(fixture).hexdigest(),'cases':results,'successful_cases':len(results)}
 (ROOT/'results.json').write_text(json.dumps(report,indent=2))
print('JOY_BLACKWELL_SMOKE_OK',flush=True)
