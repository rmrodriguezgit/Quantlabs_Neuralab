const {_electron:electron}=require('playwright');const fs=require('fs');const path=require('path');const os=require('os');const assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..');const temp=fs.mkdtempSync(path.join(os.tmpdir(),'forecast-units-'));const app=await electron.launch({executablePath:require('electron'),args:[root,'--curriculum','--user-data-dir='+path.join(temp,'profile')],env:{...process.env,FORECAST_PYTHON:process.env.FORECAST_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python'),MPLCONFIGDIR:path.join(temp,'matplotlib')}});
 try{
  const page=await app.firstWindow();const errors=[];page.on('pageerror',e=>errors.push(e.message));await page.getByRole('heading',{name:'Nueve unidades. Experimentos configurables.',exact:true}).waitFor();
  await page.getByRole('button',{name:'Crear demo OHLC sintética 2020–2025',exact:true}).click();await page.getByText('Datos cargados:',{exact:false}).waitFor({timeout:30000});
  // Inspect every infografía/preset and load its original local image.
  for(let unit=2;unit<=10;unit++){
   await page.locator('.unit-tabs button').nth(unit-2).click();await page.getByText(`Ver infografía original · U${unit}`,{exact:true}).click();const img=page.locator('.source-infographic');await img.waitFor();await img.evaluate(image=>image.decode());assert(await img.evaluate(image=>image.naturalWidth>0));await page.getByText(`Ver infografía original · U${unit}`,{exact:true}).click();
  }
  // Actual IPC execution, dates, scopes and export for each task family.
  for(const unit of [2,3,4,5,6,7,8,9,10]){
   await page.locator('.unit-tabs button').nth(unit-2).click();
   if(unit!==10)await page.getByLabel('Épocas máximas por ajuste',{exact:true}).fill('1');
   if(unit===5)await page.getByLabel('Número de trials',{exact:true}).fill('1');
   if(unit===10){await page.getByLabel('Episodios de train',{exact:true}).fill('1');await page.getByLabel('Actualizar cada N pasos',{exact:true}).fill('32')}
   await page.getByRole('button',{name:`Ejecutar unidad ${unit} →`,exact:true}).click();const results=page.locator('.unit-results');await results.waitFor({timeout:180000});await page.getByRole('button',{name:`Exportar experimento U${unit}`,exact:true}).waitFor();
   if(await page.locator('.banner.error').count())throw new Error(await page.locator('.banner.error').innerText());
   const output=path.join(temp,`unit-${unit}.json`);await app.evaluate(({dialog},filePath)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath})},output);await page.getByRole('button',{name:`Exportar experimento U${unit}`,exact:true}).click();for(let i=0;i<50&&!fs.existsSync(output);i++)await new Promise(resolve=>setTimeout(resolve,100));const data=JSON.parse(fs.readFileSync(output,'utf8'));assert.equal(data.unit,unit);assert(data.winner);assert(data.comparison.length>0);assert.equal(data.config.testEnd.slice(0,10),'2026-01-01');
   console.log('PASS UI unidad',unit,data.task,data.winner);
   if([7,8,9,10].includes(unit)){await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(root,`docs/unidad-${unit}.png`),fullPage:true})}
  }
  assert.deepEqual(errors,[]);console.log('PASS: nueve imágenes locales, presets, ejecución real U2–U10, métricas por tarea y exportaciones JSON sin errores JavaScript.');
 }finally{await app.close()}
})().catch(e=>{console.error(e);process.exit(1)});
