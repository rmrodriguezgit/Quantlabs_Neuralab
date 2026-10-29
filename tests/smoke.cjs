const {_electron:electron}=require('playwright');
const path=require('path');
const fs=require('fs');
const os=require('os');
(async()=>{
 const root=path.resolve(__dirname,'..');
 const app=await electron.launch({executablePath:require('electron'),args:[root,'--user-data-dir='+fs.mkdtempSync(path.join(os.tmpdir(),'forecast-smoke-'))],env:{...process.env,FORECAST_PYTHON:process.env.FORECAST_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python'),MPLCONFIGDIR:path.join(os.tmpdir(),'forecast-matplotlib')}});
 try{
  const page=await app.firstWindow();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.getByText('Todo comienza con una serie.').waitFor();
  await page.screenshot({path:path.join(root,'docs/inicio.png'),fullPage:true});
  await page.getByRole('button',{name:'Generar escenario',exact:true}).click();
  await page.getByText('Datos cargados:',{exact:false}).waitFor({timeout:30000});
  await page.getByRole('checkbox',{name:'ARIMA',exact:true}).uncheck();
  await page.getByRole('button',{name:'Ejecutar y comparar →',exact:true}).click();
  await page.getByText('Análisis completado',{exact:false}).waitFor({timeout:120000});
  await page.getByText('Consola comparativa',{exact:true}).waitFor();
  if(await page.locator('.banner.error').count())throw new Error(await page.locator('.banner.error').innerText());
  await page.evaluate(()=>window.scrollTo(0,0));
  await page.screenshot({path:path.join(root,'docs/dashboard.png'),fullPage:true});
  await page.getByRole('button',{name:'◇ Academia de modelos',exact:false}).click();
  await page.getByText('Comprueba lo aprendido',{exact:true}).waitFor();
  const radios=page.getByRole('radio');await radios.nth(0).check();
  await page.getByText('Correcto.',{exact:false}).waitFor();
  await page.screenshot({path:path.join(root,'docs/academia.png'),fullPage:true});
  if(errors.length)throw new Error(errors.join('\n'));
  console.log('PASS: Electron real → preload → Python; simulación, ranking, gráficas y cuestionario sin errores JS.');
 }finally{await app.close()}
})().catch(e=>{console.error(e);process.exit(1)});
