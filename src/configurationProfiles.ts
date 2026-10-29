import {Architecture,NetworkConfig,NetworkName} from './types';
import {UnitConfig,defaultUnit} from './unitConfig';
import {defaultNetwork,layer} from './networks';
export interface ConfigurationProfile {id:string;label:string;scope:string;mode:'recommended'|'free';values:Record<string,unknown>}
export interface Preset<T> {id:string;label:string;description:string;patch:Partial<T>}
// Compare only fields controlled by the profile; dates and runtime settings stay independent.
export function canonical(value:unknown):string{
 const clean=(v:unknown):unknown=>Array.isArray(v)?v.map(clean):v&&typeof v==='object'?Object.fromEntries(Object.entries(v).filter(([k,x])=>k!=='configurationProfile'&&x!==undefined).sort(([a],[b])=>a.localeCompare(b)).map(([k,x])=>[k,clean(x)])):v;
 return JSON.stringify(clean(value));
}
export function profileState(value:object,scope?:string){
 const cfg=value as {configurationProfile?:ConfigurationProfile};const p=cfg.configurationProfile;
 const same=!!p&&!!p.values&&typeof p.values==='object'&&(!scope||p.scope===scope)&&Object.keys(p.values).every(k=>canonical((value as Record<string,unknown>)[k])===canonical(p.values[k]));
 return {recommended:!!p&&p.mode==='recommended'&&same,label:p?.label,modified:!!p&&p.mode==='recommended'&&!same};
}
export function applyPreset<T extends object>(value:T,preset:Preset<T>,scope:string):T{
 const values=JSON.parse(canonical(preset.patch)) as Record<string,unknown>;
 return {...value,...preset.patch,configurationProfile:{id:preset.id,label:preset.label,scope,mode:'recommended',values}};
}
export function markFree<T extends object>(value:T):T{
 const p=(value as {configurationProfile?:ConfigurationProfile}).configurationProfile;
 return {...value,configurationProfile:{id:p?.id||'free',label:p?.label||'Personalizada',scope:p?.scope||'',mode:'free',values:p?.values||{}}};
}
export function profileForRun<T extends object>(value:T):T{
 const p=(value as {configurationProfile?:ConfigurationProfile}).configurationProfile;
 return p?{...value,configurationProfile:{...p,mode:profileState(value).recommended?'recommended':'free'}}:value;
}
function arch(widths:number[],activation='relu',lr=.001,dropout=0,conv=false):Architecture{return {inputUnits:widths[0],inputActivation:activation,layers:widths.map(units=>({...layer(activation),units,dropout,kernelSize:3,...(conv?{stride:2}: {})})),optimizer:'adam',learningRate:lr}}
const budget={epochs:1500,earlyStopping:true,patience:30,minDelta:.00001,warmupEpochs:10,batchSize:64};
export const FAMILY_NAMES:Record<string,string>={ridge:'Ridge',logistic:'Regresión logística',mlp:'MLP',cnn:'CNN 1D',rnn:'RNN',lstm:'LSTM',gru:'GRU',bilstm:'BiLSTM',bigru:'BiGRU',ae_ridge:'AE + Ridge',dae_ridge:'DAE + Ridge',pca_ridge:'PCA + Ridge',cvae:'CVAE',cgan:'cGAN denso',cdcgan:'cDCGAN 1D',dqn:'DQN',ddqn:'Double DQN'};
export function familyPatch(family:string):Partial<UnitConfig>{
 const common:Partial<UnitConfig>={...budget,featureScaler:'standard',targetScaler:'standard',projectInput:false,selectedFeatures:[...defaultUnit(2).selectedFeatures,'return_current','intraday_current','gap_current']};
 if(family==='ridge'||family==='logistic')return {...common,ridgeAlpha:1,lookback:1};
 if(family==='mlp')return {...common,projectInput:true,architecture:{...arch([32], 'relu',.001,.1),inputUnits:64},lookback:1};
 if(family==='cnn')return {...common,architecture:arch([32,64],'relu',.001,.1),lookback:20,cnnPadding:'causal'};
 if(['rnn','lstm','gru','bilstm','bigru'].includes(family))return {...common,architecture:arch([family==='rnn'?16:32],'tanh',.001,.1),lookback:family==='rnn'?10:20,recurrentModels:[family]};
 if(['ae_ridge','dae_ridge','pca_ridge'].includes(family))return {...common,architecture:arch([32]),encoder:arch([32]),decoder:arch([32]),latentDim:3,noiseStd:family==='dae_ridge'?.1:0,ridgeAlpha:1};
 if(['cvae','cgan','cdcgan'].includes(family))return {...common,batchSize:64,encoder:arch([32]),decoder:arch([32]),generator:arch(family==='cdcgan'?[32,16]:[64,32],'relu',.0002),discriminator:arch([32,64],'relu',.0002,.1),generativeModels:[family],horizon:16,conditionPeriod:20,latentDim:4,beta:.01,scenarioCount:64,conditionFeatures:['volatility','momentum','regime'],conditionEmbeddingDim:4,conditionFusion:'concat',ganUpsampling:'transpose',ganNegativeSlope:.01,ganLeaky:true,ganBatchNorm:family==='cdcgan',temporalWeight:1,scenarioRegime:'observed'};
 if(family==='dqn'||family==='ddqn')return {...common,projectInput:true,architecture:{...arch([32]),inputUnits:64},episodes:10,replayCapacity:10000,replayBatch:64,replayWarmup:128,trainEvery:4,targetSync:100,gamma:.95,epsilonStart:1,epsilonEnd:.05,epsilonDecay:5000,earlyStopping:false,costBps:5};
 return common;
}
const descriptions:Record<string,string>={mlp:'Entrada 64 → oculta 32, ReLU, Adam 0.001 y dropout 0.1: referencia densa compacta.',cnn:'20 sesiones, filtros 32 → 64, kernel 3 y padding causal para patrones locales.',rnn:'10 sesiones y 16 unidades tanh: contexto corto y pocos parámetros.',lstm:'20 sesiones, 32 unidades tanh y compuertas sigmoid para memoria temporal.',gru:'20 sesiones y 32 unidades GRU: alternativa compacta a LSTM.',bilstm:'32 unidades por sentido dentro de una ventana histórica de 20 sesiones.',bigru:'32 unidades GRU por sentido; ambas direcciones usan solo historia disponible.',ae_ridge:'32 neuronas, tres factores latentes y Ridge alpha 1 para evaluar la utilidad predictiva.',dae_ridge:'Tres factores y ruido gaussiano 0.1 en train; objetivo limpio.',pca_ridge:'Tres componentes y Ridge alpha 1 como referencia lineal.',ridge:'Alpha 1 y una sesión: referencia lineal regularizada.',logistic:'Regularización de referencia y una sesión para dirección.',cvae:'Latente 4, beta KL 0.01, ventana 16 y condiciones causales; optimizador del encoder compartido con decoder.',cgan:'G 64 → 32 y D 32 → 64, Adam 0.0002 independientes, latente 4 y condiciones previas.',cdcgan:'G con filtros 32 → 16 y D 32 → 64; kernels 3, strides 2, BatchNorm y LeakyReLU 0.01.',dqn:'Estado con posición, arquitectura 64 → 32, replay 10,000, gamma 0.95 y diez episodios.',ddqn:'Online elige y target evalúa; replay 10,000, lote 64 y sincronización cada 100 actualizaciones.'};
export function familyDescription(family:string){return descriptions[family]||'Punto de partida para comparar en validación.'}
export function unitPresets(unit:number,cfg:UnitConfig):Preset<UnitConfig>[] {
 const make=(family:string,extra:Partial<UnitConfig>={}):Preset<UnitConfig>=>({id:'unit-'+unit+'-'+family,label:FAMILY_NAMES[family]+' · recomendado',description:familyDescription(family),patch:{...familyPatch(family),...extra}});
 if(unit===2)return [make('mlp')];
 if(unit===3)return [{...make('mlp',{epochs:100,earlyStopping:false,optimizerRates:{sgd:.01,rmsprop:.001,adamw:.001}}),label:'Comparar optimizadores · presupuesto igual',description:'MLP compacta; 100 épocas completas para SGD, RMSprop y AdamW con pesos iniciales comunes.'}];
 if(unit===4)return [{...make('mlp',{epochs:200,earlyStopping:false,includeEarlyStopping:true,dropout:.3,l1:.00001,l2:.001}),label:'Regularización · comparación controlada',description:'200 épocas para variantes base, dropout, L1, L2 y BatchNorm; se incluye la variante Early Stopping.'}];
 if(unit===5){const year=Number(cfg.validationStart.slice(0,4));return [{...make('mlp',{trials:4,widths:[32,64,128],depths:[1,2,3],lrLogMin:-4,lrLogMax:-2.5,weightDecays:[0,.0001,.001],foldYears:[year-2,year-1]}),label:'HPO · búsqueda temporal compacta',description:'Cuatro trials, anchos 32/64/128, profundidades 1/2/3 y dos folds anteriores a la validación externa.'}]}
 if(unit===6)return [make('cnn')];
 if(unit===7)return ['lstm','gru','bilstm','bigru'].map(f=>make(f));
 if(unit===8)return [{...make('dae_ridge',{noiseStd:.1}),label:'PCA / AE / DAE · tres factores',description:'Encoder 32 → 3, decoder 3 → 32, ruido DAE 0.1 y Ridge alpha 1. La unidad compara los tres extractores.'}];
 if(unit===9)return ['cvae','cgan','cdcgan'].map(f=>make(f));
 return [make('ddqn')];
}
export function networkPresets(name:NetworkName):Preset<NetworkConfig>[] {
 const base=defaultNetwork(name);const family=name==='gan'?'cgan':name;const p=familyPatch(family);const a=p.architecture||base;
 const features={...base.features,lags:[1,2],ema:[10,20],rsi:true,returns:true,volatility:true};
 const patch:Partial<NetworkConfig>={featureScaler:'standard',targetScaler:'standard',targetTransform:'level',maxTrainSamples:0,...budget,...a,features,lookback:name==='rnn'?10:20,batchSize:32,validationFraction:.2};
 if(name==='gan')Object.assign(patch,p.generator,{discriminator:p.discriminator,latentDim:8,draws:64,reconstructionWeight:10});
 if(name==='dqn'){delete patch.epochs;delete patch.earlyStopping}
 const presets:Preset<NetworkConfig>[]=[{id:'network-'+name,label:(name==='gan'?'GAN de pronóstico':FAMILY_NAMES[name])+' · recomendado',description:name==='gan'?'G 64 → 32, D 32 → 64, LR 0.0002, ruido 8 y reconstrucción L1 10. Genera un valor por paso.':name==='dqn'?'Red Q 64 → 32 y ventana 20. Episodios y exploración se ajustan en RL del análisis.':familyDescription(name),patch}];
 if(['rnn','cnn','lstm','gan'].includes(name))presets.push({id:'forex-'+name,label:'Forex · '+name.toUpperCase()+' · log-retornos',description:'Punto de partida Forex: contexto 20, últimas 500 observaciones, VA interna 10%, Robust X, Standard y y objetivo log-retorno reconstruido a cotización. Compara con persistencia en los mismos folds.',patch:{...patch,epochs:300,patience:20,warmupEpochs:10,maxTrainSamples:500,validationFraction:.1,targetTransform:'log_return',featureScaler:'robust',targetScaler:'standard',inputUnits:16,layers:[{...layer(name==='rnn'||name==='lstm'?'tanh':'relu'),units:16}],learningRate:.0003}});
 return presets;
}
export function configurationSummary(cfg:Partial<UnitConfig&NetworkConfig>){
 const a=cfg.architecture||cfg;const layers=a.layers?.map(l=>l.units).join(' → ')||'—';
 return `Capas ${layers} · ${a.optimizer||'—'} · LR ${a.learningRate??'—'} · ventana ${cfg.lookback??'—'} · máximo ${cfg.epochs??'—'} épocas · ES ${cfg.earlyStopping?'activo':'apagado'}`;
}
export interface BankSetup {cfg:UnitConfig;families:string[];trials:number;widths:number[];depths:number[];lookbacks:number[];latentDims:number[];low:number;high:number;optimizers:string[];dropouts:number[];configurationProfile?:ConfigurationProfile}
export function bankPresets(mode:string,families:string[],cfg:UnitConfig):Preset<BankSetup>[] {
 return families.map(family=>{
  const patch=familyPatch(family);const recurrent=['rnn','lstm','gru','bilstm','bigru'].includes(family);const generation=['cvae','cgan','cdcgan'].includes(family);const lr=generation&&family!=='cvae'?Math.log10(.0002):-3;
  return {id:'bank-'+mode+'-'+family,label:FAMILY_NAMES[family]+' · búsqueda recomendada',description:`${FAMILY_NAMES[family]}: tres variantes con anchos 32/64 (RNN: 16/32), Adam y LR en torno a ${generation&&family!=='cvae'?'0.0002':'0.001'}. Las capas se muestrean del espacio mostrado; no son una arquitectura fija.${['ridge','logistic','pca_ridge'].includes(family)?' El ajuste lineal no usa épocas.':''}`,patch:{cfg:{...cfg,...patch},families:[family],trials:3,widths:family==='rnn'?[16,32]:[32,64],depths:recurrent?[1]:generation?[1,2]:[1,2],lookbacks:recurrent||family==='cnn'?[10,20]:[1],latentDims:[2,3],low:lr-.3,high:lr+.3,optimizers:['adam'],dropouts:[0,.1]}};
 });
}
export function unitSummary(unit:number,cfg:UnitConfig){
 if(unit===9)return `${cfg.generativeModels.join(' / ')} · G ${cfg.generator.layers.map(l=>l.units).join(' → ')} ${cfg.generator.optimizer} LR ${cfg.generator.learningRate} · D ${cfg.discriminator.layers.map(l=>l.units).join(' → ')} LR ${cfg.discriminator.learningRate} · latente ${cfg.latentDim} · ${cfg.conditionFeatures.join(', ')} · ${cfg.epochs} épocas máximas · ES ${cfg.earlyStopping?'activo':'apagado'}`;
 if(unit===10)return `Double DQN · ${cfg.architecture.inputUnits} → ${cfg.architecture.layers.map(l=>l.units).join(' → ')} · ${cfg.episodes} episodios · replay ${cfg.replayCapacity} · lote ${cfg.replayBatch} · gamma ${cfg.gamma}`;
 if(unit===8)return `PCA / AE / DAE · ${cfg.latentDim} factores · encoder ${cfg.encoder.layers.map(l=>l.units).join(' → ')} · ruido ${cfg.noiseStd} · ${cfg.epochs} épocas máximas · ES ${cfg.earlyStopping?'activo':'apagado'}`;
 return configurationSummary(cfg)+(unit===5?` · ${cfg.trials} trials · folds ${cfg.foldYears.join(', ')}`:'');
}
