import argparse,pathlib,json,hashlib,tarfile
parser=argparse.ArgumentParser()
parser.add_argument('--preview-side',type=int,default=0,
                    help='Optional thumbnail size; originals remain unchanged on Global')
args=parser.parse_args()
assert args.preview_side>=0
root=pathlib.Path('/opt/moss-bench');out=root/'results'
status=json.loads((root/'status.json').read_text());print('STATUS',json.dumps(status))
assert status['state']=='complete','suite has not completed'
store=pathlib.Path('/workspace/benchmarks/2026-10-19-moss')/status['label']
manifest=json.loads((store/'sha256.json').read_text())
for name,digest in manifest.items():
 assert hashlib.sha256((store/name).read_bytes()).hexdigest()==digest,(name,'global mismatch')
assert len([n for n in manifest if n.endswith('.png')])>0
verification={'files':len(manifest),'global_path':str(store),'sha256_verified':True,'output_images':len([n for n in manifest if n.endswith('.png')])}
(out/'global-verification.json').write_text(json.dumps(verification,indent=2))
(out/'global-sha256.json').write_text(json.dumps(manifest,indent=2))
# Metadata includes hashes of all generated outputs; originals remain on Global.
with tarfile.open(root/'metadata.tar.gz','w:gz') as tar:
 for f in out.iterdir():
  if f.is_file() and f.suffix!='.png':tar.add(f,arcname=f.name)
 tar.add(root/'status.json',arcname='status.json')
preview=[]
for summary in json.loads((out/'summary.json').read_text()):
 side=summary['confirmed_side_with_headroom']
 if side:preview.append(out/f'moss_image.{summary["variant"]}-confirm-{side}-{side}-r0.png')
with tarfile.open(root/'previews.tar.gz','w:gz') as tar:
 for f in preview:
  if not f.exists():continue
  if args.preview_side:
   from PIL import Image
   with Image.open(f) as image:
    image=image.convert('RGB');image.thumbnail((args.preview_side,args.preview_side))
    thumbnail=root/(f.stem+'.preview.jpg');image.save(thumbnail,quality=90)
   tar.add(thumbnail,arcname=thumbnail.name)
  else:tar.add(f,arcname=f.name)
print('GLOBAL_VERIFIED',json.dumps(verification))
print('ARCHIVE', (root/'metadata.tar.gz').stat().st_size)
print('PREVIEWS',[p.name for p in preview])
