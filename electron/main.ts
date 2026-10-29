import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs/promises';
let win: BrowserWindow;
let configuredPython: string | undefined;
const children = new Set<ReturnType<typeof spawn>>();
const root = () => app.isPackaged ? process.resourcesPath : path.resolve(__dirname, '..');
function engine(request: unknown): Promise<any> {
  const base = root();
  const runtime = app.isPackaged ? 'python-runtime' : '.venv';
  const python = process.env.FORECAST_PYTHON || configuredPython || path.join(base, runtime, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const input = JSON.stringify(request);
  if (Buffer.byteLength(input) > 20_000_000) return Promise.reject(new Error('Solicitud superior a 20 MB.'));
  const settings = request as {config?:{timeoutMinutes?:number};options?:{timeoutMinutes?:number}};
  const timeoutMinutes=Number(settings.config?.timeoutMinutes ?? settings.options?.timeoutMinutes ?? 60);
  if(!Number.isInteger(timeoutMinutes)||timeoutMinutes<1||timeoutMinutes>240)return Promise.reject(new Error('Tiempo de ejecución: entre 1 y 240 minutos.'));
  return new Promise((resolve, reject) => {
    const child = spawn(python, [path.join(base,'python','engine.py')], { stdio:['pipe','pipe','pipe'], windowsHide:true, env:{...process.env,PYTHONUTF8:'1',TF_CPP_MIN_LOG_LEVEL:'3'} });
    children.add(child);
    let output = '', diagnostic = '', settled = false;
    const finish = (err?: Error, value?: unknown) => { if(settled) return; settled=true; clearTimeout(timer); children.delete(child); err ? reject(err) : resolve(value); };
    const timer = setTimeout(() => { child.kill(); finish(new Error('El cálculo excedió el tiempo configurado. Aumenta los minutos de ejecución o reduce el presupuesto.')); }, timeoutMinutes*60_000);
    child.stdout.on('data', b => { output += b.toString(); if(output.length>60_000_000){ child.kill(); finish(new Error('Respuesta demasiado grande.')); } });
    let stderrBuffer='';
    child.stderr.on('data', b => { diagnostic=(diagnostic+b.toString()).slice(-4000);stderrBuffer+=b.toString();let at:number;while((at=stderrBuffer.indexOf('\n'))>=0){const line=stderrBuffer.slice(0,at);stderrBuffer=stderrBuffer.slice(at+1);if(line.startsWith('LAB_PROGRESS ')){try{const value=JSON.parse(line.slice(13));if(win&&!win.isDestroyed())win.webContents.send('engine:progress',value);}catch{}}}if(stderrBuffer.length>100000)stderrBuffer=stderrBuffer.slice(-10000); });
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
function trusted(event: Electron.IpcMainInvokeEvent) {
  if (event.sender !== win.webContents || event.senderFrame !== win.webContents.mainFrame) throw new Error('Origen IPC no autorizado');
}
ipcMain.handle('engine:run', async (event, request) => {
  trusted(event);
  if (!request || !['simulate','finance','analyze','capabilities','unit_lab','demo_ohlc','emulate'].includes(request.action)) throw new Error('Operación no permitida');
  if(request.action==='emulate'){const artifactRoot=path.join(app.getPath('userData'),'forecast-artifacts');await fs.mkdir(artifactRoot,{recursive:true});return engine({...request,artifactRoot});}
  return engine(request);
});
ipcMain.handle('files:import', async event => {
  trusted(event);
  const result=await dialog.showOpenDialog(win,{properties:['openFile','multiSelections'],filters:[{name:'Datos',extensions:['csv','xlsx']}]});
  if(result.canceled) return [];
  return engine({action:'import',paths:result.filePaths});
});
ipcMain.handle('files:save', async (event, request) => {
  trusted(event);
  if(!request || !['csv','xlsx','json'].includes(request.format)) throw new Error('Formato no permitido');
  const format=request.format as string;
  const result=await dialog.showSaveDialog(win,{defaultPath:`forecast-lab.${format}`,filters:[{name:format.toUpperCase(),extensions:[format]}]});
  if(result.canceled || !result.filePath) return false;
  if(format==='json') await fs.writeFile(result.filePath,JSON.stringify(request.data,null,2),'utf8');
  else await engine({action:'export',path:result.filePath,format,dataset:request.data});
  return true;
});
ipcMain.handle('artifacts:save',async(event,id:string)=>{
  trusted(event);if(typeof id!=='string'||!(/^[a-f0-9]{32}$/).test(id))throw new Error('Paquete no válido.');
  const source=path.join(app.getPath('userData'),'forecast-artifacts',id+'.zip');await fs.stat(source);
  const result=await dialog.showSaveDialog(win,{defaultPath:'forecast-model-'+id.slice(0,8)+'.zip',filters:[{name:'Paquete de inferencia',extensions:['zip']}]});
  if(result.canceled||!result.filePath)return false;await fs.copyFile(source,result.filePath);return true;
});
ipcMain.handle('artifacts:bundle',async(event,ids:string[],report:unknown)=>{
  trusted(event);if(!Array.isArray(ids)||ids.length>20||!ids.length||ids.some(id=>typeof id!=='string'||!(/^[a-f0-9]{32}$/).test(id)))throw new Error('Paquetes no válidos.');
  return engine({action:'bundle_artifacts',ids,report,artifactRoot:path.join(app.getPath('userData'),'forecast-artifacts')});
});
app.whenReady().then(async () => {
  if (app.isPackaged && !process.env.FORECAST_PYTHON) {
    const configPath=path.join(app.getPath('userData'),'runtime.json');
    try { configuredPython=JSON.parse(await fs.readFile(configPath,'utf8')).python; } catch {}
    if (!configuredPython) {
      const choice=await dialog.showOpenDialog({title:'Selecciona el ejecutable Python de tu .venv con las dependencias instaladas',properties:['openFile','showHiddenFiles']});
      if (choice.canceled) { app.quit(); return; }
      configuredPython=choice.filePaths[0];
      await fs.mkdir(path.dirname(configPath),{recursive:true});
      await fs.writeFile(configPath,JSON.stringify({python:configuredPython}),'utf8');
    }
  }
  const create = () => {
    win=new BrowserWindow({width:1480,height:980,minWidth:1000,minHeight:720,backgroundColor:'#f2f5f8',webPreferences:{preload:path.join(__dirname,'preload.js'),contextIsolation:true,nodeIntegration:false,sandbox:true}});
    win.webContents.setWindowOpenHandler(() => ({action:'deny'}));
    win.webContents.on('will-navigate', event => event.preventDefault());
    win.loadFile(path.join(__dirname,'..','dist','index.html'),{hash:process.argv.includes('--emulator')?'emulator':process.argv.includes('--curriculum')?'curriculum':process.argv.includes('--neural-studio')?'networks':''});
  };
  create(); app.on('activate',()=>{if(BrowserWindow.getAllWindows().length===0) create();});
});
app.on('before-quit',()=>children.forEach(c=>c.kill()));
app.on('window-all-closed',()=>{if(process.platform!=='darwin') app.quit();});
