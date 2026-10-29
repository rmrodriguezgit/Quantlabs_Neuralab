import {Architecture,FeatureConfig,NetworkConfig,NetworkConfigs,NetworkName,LayerConfig} from './types';
export const NETWORK_NAMES:NetworkName[]=['rnn','cnn','lstm','dqn','gan'];
export const ACTIVATIONS=['linear','relu','tanh','sigmoid','elu','selu','softplus','swish','leaky_relu'];
export const OPTIMIZERS=['adam','adamw','rmsprop','sgd','adamax','nadam','adagrad'];
export const NETWORK_STORAGE='forecast-lab-networks-v1';
export function layer(activation='relu'):LayerConfig{return {units:24,activation,recurrentActivation:'sigmoid',dropout:0,kernelSize:3}}
export function architecture(activation='relu'):Architecture{return {inputUnits:16,inputActivation:'linear',layers:[layer(activation)],optimizer:'adam',learningRate:.001}}
export function defaultNetwork(name:NetworkName):NetworkConfig{return {featureScaler:'standard',targetScaler:'standard',targetTransform:'level',maxTrainSamples:0,...architecture(name==='rnn'||name==='lstm'?'tanh':'relu'),lookback:8,epochs:1500,earlyStopping:true,patience:30,minDelta:.00001,warmupEpochs:10,validationFraction:.2,batchSize:32,features:{lags:[1,2,3],ema:[],sma:[],rsi:false,rsiPeriod:14,macd:false,macdFast:12,macdSlow:26,macdSignal:9,returns:false,volatility:false,volatilityPeriod:20},discriminator:architecture(),latentDim:8,draws:32,reconstructionWeight:10}}
export function loadNetworks():NetworkConfigs{
 let saved:Partial<NetworkConfigs>={};try{saved=JSON.parse(localStorage.getItem(NETWORK_STORAGE)||'{}')}catch{}
 return Object.fromEntries(NETWORK_NAMES.map(name=>{const d=defaultNetwork(name),s=saved?.[name];if(!s||!Array.isArray(s.layers)||!s.layers.length||s.layers.length>5)return [name,d];return [name,{...d,...s,features:{...d.features,...s.features},discriminator:{...d.discriminator,...s.discriminator}}]})) as NetworkConfigs;
}
export function featureCount(f:FeatureConfig){return 1+f.lags.length+f.ema.length+f.sma.length+(f.rsi?1:0)+(f.macd?3:0)+(f.returns?1:0)+(f.volatility?1:0)}
export function warmup(f:FeatureConfig){return Math.max(0,...f.lags,...f.ema.map(x=>x-1),...f.sma.map(x=>x-1),f.rsi?f.rsiPeriod:0,f.macd?f.macdSlow+f.macdSignal-2:0,f.returns?1:0,f.volatility?f.volatilityPeriod:0)}
