"""Verifica el intérprete aislado y las importaciones antes de abrir Electron."""
import importlib,json,sys
from pathlib import Path

def main():
 if not (3,10)<=sys.version_info[:2]<=(3,12):raise SystemExit('Se requiere Python 3.10–3.12; recrea .venv con un intérprete compatible.')
 if sys.prefix==sys.base_prefix:raise SystemExit('Ejecuta este diagnóstico dentro de .venv.')
 modules=['numpy','pandas','scipy','statsmodels.api','arch','sklearn','yfinance','openpyxl']
 if len(sys.argv)>1 and 'advanced' in Path(sys.argv[1]).name:modules+=['tensorflow','prophet','gymnasium']
 errors=[]
 for name in modules:
  try:importlib.import_module(name)
  except Exception as e:errors.append(f'{name}: {e}')
 if errors:raise SystemExit('Entorno incompleto o incompatible:\n'+'\n'.join(errors))
 print(json.dumps({'status':'ok','python':sys.version.split()[0],'environment':sys.prefix,'imports':modules}))
if __name__=='__main__':main()
