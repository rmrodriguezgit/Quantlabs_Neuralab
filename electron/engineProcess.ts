import {spawn} from 'node:child_process';
import path from 'node:path';
export class PythonEngine {
 private children=new Set<ReturnType<typeof spawn>>();
 constructor(private runtime:()=>{base:string;python:string},private progress:(value:unknown)=>void){}
 stopAll(){this.children.forEach(c=>c.kill());}
 run(request: unknown): Promise<any> {
  const {base,python}=this.runtime();
  const input = JSON.stringify(request);
  if (Buffer.byteLength(input) > 20_000_000) return Promise.reject(new Error('Solicitud superior a 20 MB.'));
  const settings = request as {config?:{timeoutMinutes?:number};options?:{timeoutMinutes?:number}};
  const timeoutMinutes=Number(settings.config?.timeoutMinutes ?? settings.options?.timeoutMinutes ?? 60);
  if(!Number.isInteger(timeoutMinutes)||timeoutMinutes<1||timeoutMinutes>240)return Promise.reject(new Error('Tiempo de ejecución: entre 1 y 240 minutos.'));
  return new Promise((resolve, reject) => {
    const child = spawn(python, [path.join(base,'python','engine_runner.py')], { stdio:['pipe','pipe','pipe'], windowsHide:true, env:{...process.env,PYTHONUTF8:'1',TF_CPP_MIN_LOG_LEVEL:'3'} });
    this.children.add(child);
    let output = '', diagnostic = '', settled = false;
    const finish = (err?: Error, value?: unknown) => { if(settled) return; settled=true; clearTimeout(timer); this.children.delete(child); err ? reject(err) : resolve(value); };
    const timer = setTimeout(() => { child.kill(); finish(new Error('El cálculo excedió el tiempo configurado. Aumenta los minutos de ejecución o reduce el presupuesto.')); }, timeoutMinutes*60_000);
    child.stdout.on('data', b => { output += b.toString(); if(output.length>60_000_000){ child.kill(); finish(new Error('Respuesta demasiado grande.')); } });
    let stderrBuffer='';
    child.stderr.on('data', b => { diagnostic=(diagnostic+b.toString()).slice(-4000);stderrBuffer+=b.toString();let at:number;while((at=stderrBuffer.indexOf('\n'))>=0){const line=stderrBuffer.slice(0,at);stderrBuffer=stderrBuffer.slice(at+1);if(line.startsWith('LAB_PROGRESS ')){try{const value=JSON.parse(line.slice(13));this.progress(value);}catch{}}}if(stderrBuffer.length>100000)stderrBuffer=stderrBuffer.slice(-10000); });
    child.stdin.on('error', e => finish(e));
    child.on('error', e => finish(new Error(`No se pudo iniciar Python: ${e.message}. Ejecuta el arrancador.`)));
    child.on('close', code => {
      if(settled) return;
      try { const response=JSON.parse(output); if(!response.ok) finish(new Error(response.error || 'Error del motor')); else finish(undefined,response.data); }
      catch { finish(new Error(`El motor terminó (${code}): ${diagnostic || 'respuesta JSON inválida'}`)); }
    });
    child.stdin.end(input);
  });
}
}
