const {_electron:electron}=require('playwright');
const path=require('path');const fs=require('fs');const os=require('os');const assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..');const temp=fs.mkdtempSync(path.join(os.tmpdir(),'forecast-neural-'));const report=path.join(temp,'report.json');
 const app=await electron.launch({executablePath:require('electron'),args:[root,'--user-data-dir='+path.join(temp,'profile')],env:{...process.env,FORECAST_PYTHON:process.env.FORECAST_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python'),MPLCONFIGDIR:path.join(temp,'matplotlib')}});
 try{
  const page=await app.firstWindow();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.getByRole('button',{name:'Generar escenario',exact:true}).click();await page.getByText('Datos cargados:',{exact:false}).waitFor({timeout:30000});
  await page.getByRole('button',{name:'Laboratorio neuronal',exact:false}).click();await page.getByRole('button',{name:'GAN condicional',exact:true}).click();
  await page.getByRole('checkbox',{name:'Comparar GAN condicional en el próximo análisis',exact:true}).check();
  await page.getByLabel('Ventana de entrada',{exact:true}).fill('4');await page.getByLabel('Épocas de entrenamiento',{exact:true}).fill('1');await page.getByLabel('Tamaño de lote',{exact:true}).fill('16');await page.getByLabel('Dimensión del ruido latente').fill('4');await page.getByLabel('Escenarios por paso').fill('8');await page.getByLabel('Peso de reconstrucción L1').fill('2');
  const G=page.locator('.architecture-card').filter({has:page.getByRole('heading',{name:'Generador GAN',exact:true})});
  const D=page.locator('.architecture-card').filter({has:page.getByRole('heading',{name:'Discriminador GAN',exact:true})});
  await G.getByRole('button',{name:'+ Agregar capa intermedia',exact:true}).click();await G.getByRole('button',{name:'+ Agregar capa intermedia',exact:true}).click();
  await G.getByLabel('Generador GAN: neuronas de entrada',{exact:true}).fill('9');await G.getByLabel('Activación de entrada',{exact:true}).selectOption('tanh');await G.getByLabel('Generador GAN: optimizador',{exact:true}).selectOption('rmsprop');
  for(let i=1;i<=3;i++)await G.getByLabel(`Generador GAN: neuronas capa ${i}`,{exact:true}).fill(String(10-i*2));
  for(let i=0;i<4;i++)await D.getByRole('button',{name:'+ Agregar capa intermedia',exact:true}).click();assert(await D.getByRole('button',{name:'+ Agregar capa intermedia',exact:true}).isDisabled());
  for(let i=5;i>=3;i--)await D.getByRole('button',{name:`Discriminador GAN: eliminar capa ${i}`,exact:true}).click();
  await D.getByLabel('Discriminador GAN: optimizador',{exact:true}).selectOption('sgd');
  const periods=[['Lags adicionales','1, 3'],['Periodos EMA','5, 12'],['Periodos SMA','5']];for(const [name,value] of periods){await page.getByLabel(name,{exact:true}).fill(value);await page.getByLabel(name,{exact:true}).press('Tab')}
  await page.getByRole('checkbox',{name:'RSI · suavizado Wilder',exact:true}).check();await page.getByLabel('Periodo RSI',{exact:true}).fill('5');
  await page.getByRole('checkbox',{name:'MACD + señal + histograma',exact:true}).check();await page.getByRole('checkbox',{name:'Variación relativa',exact:true}).check();await page.getByRole('checkbox',{name:'Volatilidad de cambios',exact:true}).check();await page.getByLabel('Periodo de volatilidad',{exact:true}).fill('5');
  await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(root,'docs/redes-configurables.png'),fullPage:true});
  await page.getByRole('button',{name:'Aplicar configuración y volver al análisis →',exact:true}).click();
  for(const name of ['Persistencia (base)','Regresión lineal','ARIMA'])await page.getByRole('checkbox',{name,exact:true}).uncheck();
  await page.getByLabel('Ventanas',{exact:true}).fill('1');await page.getByLabel('Pasos por validación',{exact:true}).fill('2');await page.getByLabel('Pasos futuros',{exact:true}).fill('3');
  await page.getByRole('button',{name:'Ejecutar y comparar →',exact:true}).click();await page.getByText('Análisis completado · GAN condicional',{exact:true}).waitFor({timeout:180000});
  await page.getByRole('heading',{name:'Arquitectura ejecutada',exact:true}).waitFor();await page.getByRole('heading',{name:'Entrenamiento adversarial · pérdidas por época',exact:true}).waitFor();
  await app.evaluate(({dialog},filePath)=>{dialog.showSaveDialog=async()=>({canceled:false,filePath})},report);
  await page.getByRole('button',{name:'Exportar reporte JSON',exact:true}).click();await page.waitForFunction(()=>!document.body.textContent.includes('El motor está trabajando.'));
  for(let i=0;i<20&&!fs.existsSync(report);i++)await new Promise(r=>setTimeout(r,100));
  const saved=JSON.parse(fs.readFileSync(report,'utf8'));const cfg=saved.result.experiment.options.networks.gan;
  assert.equal(cfg.layers.length,3);assert.equal(cfg.inputUnits,9);assert.equal(cfg.inputActivation,'tanh');assert.equal(cfg.optimizer,'rmsprop');assert.equal(cfg.discriminator.layers.length,2);assert.equal(cfg.discriminator.optimizer,'sgd');assert.deepEqual(cfg.features.ema,[5,12]);assert.equal(cfg.features.macd,true);assert.equal(cfg.features.rsi,true);
  assert.equal(saved.result.future.network.discriminator.layers.at(-1).activation,'sigmoid');assert.equal(saved.result.future.network.layers.at(-1).activation,'linear');assert.equal(saved.result.future.ganLoss.length,1);assert.equal(saved.result.future.prediction.length,3);
  const stored=await page.evaluate(()=>JSON.parse(localStorage.getItem('forecast-lab-networks-v1')));assert.equal(stored.gan.inputUnits,9);
  await page.evaluate(()=>window.scrollTo(0,0));await page.screenshot({path:path.join(root,'docs/gan-dashboard.png'),fullPage:true});
  assert.deepEqual(errors,[]);console.log('PASS: editor 1–5 capas, entrada/activaciones/optimizadores independientes, indicadores, persistencia, entrenamiento GAN real, métricas, gráficas y exportación reproducible.');
 }finally{await app.close()}
})().catch(e=>{console.error(e);process.exit(1)});
