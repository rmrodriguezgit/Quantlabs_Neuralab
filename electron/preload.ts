import { contextBridge, ipcRenderer } from 'electron';
contextBridge.exposeInMainWorld('lab', {
  onProgress: (callback:(value:unknown)=>void) => {const listener=(_event:unknown,value:unknown)=>callback(value);ipcRenderer.on('engine:progress',listener);return ()=>ipcRenderer.removeListener('engine:progress',listener);},
  saveArtifact: (id:string) => ipcRenderer.invoke('artifacts:save',id),
  bundleArtifacts: (ids:string[],report:unknown) => ipcRenderer.invoke('artifacts:bundle',ids,report),
  run: (request: unknown) => ipcRenderer.invoke('engine:run', request),
  importFiles: () => ipcRenderer.invoke('files:import'),
  save: (request: unknown) => ipcRenderer.invoke('files:save', request)
});
