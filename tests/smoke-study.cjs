const {_electron:electron}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('fs'),path=require('path'),os=require('os');
(async()=>{
 const root=path.resolve(__dirname,'..'),temp=fs.mkdtempSync(path.join(os.tmpdir(),'quantlabs-study-'));
 const csv=path.join(temp,'study.csv'),output=path.join(temp,'study.json');
 const lines=['date,Close,Other'];for(let i=0;i<240;i++)lines.push(`${new Date(Date.UTC(2024,0,i+1)).toISOString().slice(0,10)},${100+i*.03+Math.sin(i*2*Math.PI/7)},${80+i*.02+Math.cos(i/5)}`);fs.writeFileSync(csv,lines.join('\n'));
 const app=await electron.launch({executablePath:require('electron'),args:[root,'--study','--user-data-dir='+path.join(temp,'profile')],env:process.env});
 try{
  const page=await app.firstWindow(),errors=[];page.on('pageerror',e=>errors.push(e.message));const area=page.locator('.study');
  await area.getByLabel('Módulo de estudio',{exact:true}).locator('option').nth(8).waitFor({state:'attached'});assert.equal(await area.getByLabel('Módulo de estudio',{exact:true}).locator('option').count(),9);
  await area.getByRole('button',{name:'Marco teórico y cuestionario',exact:true}).click();assert.equal(await area.locator('.academy fieldset').count(),5);await area.locator('.academy input[type=radio]').first().check();await area.getByText('Volver a práctica',{exact:true}).click();
  await app.evaluate(({dialog},file)=>{dialog.showOpenDialog=async()=>({canceled:false,filePaths:[file]})},csv);
  await area.getByRole('button',{name:'Importar CSV / Excel · varios archivos',exact:true}).click();await page.getByText('Datos cargados:',{exact:false}).waitFor({timeout:30000});
  // Import defaults to business, monthly inference; use explicit financial daily declaration in existing source controls.
  await page.locator('.sidebar').getByRole('button',{name:'Análisis y simulación',exact:false}).click();await page.getByLabel('Tipo de datos',{exact:true}).selectOption('financial');await page.getByLabel('Frecuencia declarada',{exact:true}).selectOption('daily');await page.locator('.sidebar').getByRole('button',{name:'Estudio individual · V1–V3',exact:false}).click();
  await area.getByLabel('Máximo de épocas · redes',{exact:true}).fill('1');await area.getByLabel('Épocas de calentamiento',{exact:true}).fill('0');await area.getByLabel('Folds VA externos',{exact:true}).fill('1');
  const run=area.getByRole('button',{name:'Ejecutar V1 → V2 → V3',exact:true});assert(await run.isDisabled());await area.getByRole('button',{name:'Revisar TR / VA / TE',exact:true}).click();await run.waitFor();await page.waitForFunction(()=>!document.querySelector('.study button.primary').disabled);
  await area.getByLabel('Periodo estacional',{exact:true}).fill('5');assert(await run.isDisabled());await area.getByRole('button',{name:'Revisar TR / VA / TE',exact:true}).click();await page.waitForFunction(()=>!document.querySelector('.study button.primary').disabled);await run.click();await area.getByRole('button',{name:'Descargar estudio JSON',exact:true}).waitFor({timeout:120000});
  await app.evaluate(({dialog},file)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath:file})},output);await area.getByRole('button',{name:'Descargar estudio JSON',exact:true}).click();await page.waitForFunction(()=>!document.querySelector('header .activity-status').getAttribute('aria-busy')||document.querySelector('header .activity-status').getAttribute('aria-busy')==='false');const r=JSON.parse(fs.readFileSync(output));assert.equal(r.reports[0].comparisons.length,3);assert.equal(r.failures.length,0);assert(r.reports[0].test.metrics);await area.screenshot({path:path.join(root,'docs/estudio-individual-ui.png')});
  await area.getByLabel('Módulo de estudio',{exact:true}).selectOption('lstm');
  await area.getByText('Ajuste libre · parámetros JSON de LSTM',{exact:false}).click();
  for(const v of [1,2,3])await area.getByLabel('Parámetros V'+v,{exact:true}).fill(JSON.stringify({units:[4],lookback:5,dropout:0}));
  await area.getByRole('checkbox',{name:'Permitir que gane persistencia',exact:false}).uncheck();
  await area.getByRole('button',{name:'Revisar TR / VA / TE',exact:true}).click();await page.waitForFunction(()=>!document.querySelector('.study button.primary').disabled);await run.click();
  await page.waitForFunction(()=>document.querySelector('header .activity-status').getAttribute('aria-busy')==='false',{},{timeout:120000});
  await area.getByRole('button',{name:'Descargar estudio JSON',exact:true}).click();await page.waitForFunction(()=>document.querySelector('header .activity-status').getAttribute('aria-busy')==='false');
  const neural=JSON.parse(fs.readFileSync(output));assert.equal(neural.reports[0].comparisons.length,3);assert.equal(neural.failures.length,0);assert(neural.reports[0].comparisons.find(v=>v.variant===3).folds[0].metadata.layers.some(l=>l.type==='MultiHeadAttention'));await page.screenshot({path:path.join(root,'docs/estudio-configuracion-ui.png')});
  await page.locator('.sidebar').getByRole('button',{name:'Academia de modelos',exact:false}).click();await page.locator('.sidebar').getByRole('button',{name:'Estudio individual · V1–V3',exact:false}).click();await area.getByRole('button',{name:'Descargar estudio JSON',exact:true}).waitFor();assert.deepEqual(errors,[]);
  console.log('PASS: nueve módulos, teoría y cinco preguntas, revisión invalidada al editar, tres regresiones y tres LSTM reales con atención, TE, JSON y resultado conservado.');
 }finally{await app.close()}
})().catch(e=>{console.error(e);process.exit(1)});
