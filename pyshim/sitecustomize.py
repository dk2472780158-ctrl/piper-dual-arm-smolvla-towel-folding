import transformers.utils as u
def _tcc(c,m=None):
 if not c: raise ValueError(m)
if not hasattr(u,'torch_compilable_check'):
 u.torch_compilable_check=_tcc
import huggingface_hub as h
def _sb(*a,**k):
 raise NotImplementedError('stub')
if not hasattr(h,'sync_bucket'):
 h.sync_bucket=_sb

def _h():
 import os
 from transformers import AutoModelForImageTextToText as A
 o=A.from_pretrained
 def p(m,*a,**k):
  if isinstance(m,str) and m.startswith('HuggingFaceTB/'):
   l='/workspace/models/'+m[13:]
   if os.path.isdir(l):m=l
  return o(m,*a,**k)
 A.from_pretrained=p
_h()

def _h2():
 import os
 from transformers import AutoProcessor
 o=AutoProcessor.from_pretrained
 def p(m,*a,**k):
  if isinstance(m,str) and m.startswith('HuggingFaceTB/'):
   l='/workspace/models/'+m[13:]
   if os.path.isdir(l):m=l
  return o(m,*a,**k)
 AutoProcessor.from_pretrained=p
_h2()
