"""Entrada JSON estable del orquestador; adapta acciones existentes y módulos nuevos."""
import contextlib,json,sys
from engine import dispatch,clean

def main():
    try:
        raw=sys.stdin.read(20_000_001)
        if len(raw)>20_000_000:raise ValueError('Solicitud demasiado grande.')
        req=json.loads(raw)
        with contextlib.redirect_stdout(sys.stderr):data=dispatch(req)
        response={'ok':True,'data':clean(data)}
    except Exception as e:response={'ok':False,'error':f'{type(e).__name__}: {e}'}
    sys.stdout.write(json.dumps(response,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
