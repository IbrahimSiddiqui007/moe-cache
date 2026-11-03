from PIL import Image
import os, sys
d=os.path.expanduser('~/.cache/moe-desktop-tmp/ic'); out=sys.argv[1]
tiles=[]
for name,show in [('s16',16),('s32',32),('f256',256),('f1024',512)]:
    im=Image.open(f'{d}/{name}.png').convert('RGBA')
    if show!=im.width: im=im.resize((show,show),Image.LANCZOS)
    tiles.append(im)
W=40+sum(t.width+40 for t in tiles); H=512+80
sheet=Image.new('RGB',(W,H*2))
for row,bg in enumerate([(246,248,250),(22,27,34)]):
    sheet.paste(Image.new('RGB',(W,H),bg),(0,row*H)); x=40
    for im in tiles:
        sheet.paste(im,(x,row*H+40+(512-im.height)//2),im); x+=im.width+40
sheet.save(out)
