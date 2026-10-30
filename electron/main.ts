import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import {PythonEngine} from './engineProcess';
import path from 'node:path';
import fs from 'node:fs/promises';
let win: BrowserWindow;
let configuredPython: string | undefined;
const root = () => app.isPackaged ? process.resourcesPath : path.resolve(__dirname, '..');
const manager=new PythonEngine(()=>{const base=root();const runtime=app.isPackaged?'python-runtime':'.venv';return {base,python:process.env.FORECAST_PYTHON||configuredPython||path.join(base,runtime,process.platform==='win32'?'Scripts/python.exe':'bin/python')}} ,value=>{if(win&&!win.isDestroyed())win.webContents.send('engine:progress',value)});
const engine=(request:unknown)=>manager.run(request);
function trusted(event: Electron.IpcMainInvokeEvent) {
  if (event.sender !== win.webContents || event.senderFrame !== win.webContents.mainFrame) throw new Error('Origen IPC no autorizado');
}
ipcMain.handle('engine:run', async (event, request) => {
  trusted(event);
  if (!request || !['simulate','finance','analyze','capabilities','unit_lab','demo_ohlc','emulate','preview_experiment','strategy_preview','strategy_search','study_catalog','study_preview','study_run'].includes(request.action)) throw new Error('Operación no permitida');
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
  const storedUserData=app.getPath('userData');
  app.setName('Quantlabs Neurolab');
  app.setPath('userData',storedUserData);
  const iconPath=path.join(__dirname,'..','dist','icon.png');
  if(process.platform==='darwin')app.dock?.setIcon(iconPath);
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
    win=new BrowserWindow({title:'Quantlabs Neurolab · v'+app.getVersion(),icon:iconPath,width:1480,height:980,minWidth:1000,minHeight:720,backgroundColor:'#f2f5f8',webPreferences:{preload:path.join(__dirname,'preload.js'),contextIsolation:true,nodeIntegration:false,sandbox:true}});
    win.webContents.setWindowOpenHandler(() => ({action:'deny'}));
    win.webContents.on('will-navigate', event => event.preventDefault());
    win.loadFile(path.join(__dirname,'..','dist','index.html'),{hash:process.argv.includes('--study')?'study':process.argv.includes('--emulator')?'emulator':process.argv.includes('--curriculum')?'curriculum':process.argv.includes('--neural-studio')?'networks':''});
  };
  create(); app.on('activate',()=>{if(BrowserWindow.getAllWindows().length===0) create();});
});
app.on('before-quit',()=>manager.stopAll());
app.on('window-all-closed',()=>{if(process.platform!=='darwin') app.quit();});
