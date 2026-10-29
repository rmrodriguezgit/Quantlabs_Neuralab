"""U10: Double DQN, posiciones ejecutadas a la siguiente apertura y costes."""
from collections import deque
import numpy as np
from features import bounded_int
from neural import real,optimizer
from financial_lab import panel,scale_panel,split_summary,trade_path


def double_targets(rewards,done,online_next,target_next,gamma):
    chosen=np.argmax(online_next,axis=1)
    return np.asarray(rewards)+gamma*(1-np.asarray(done))*target_next[np.arange(len(chosen)),chosen]


def session_return(old,new,row,cost_bps,liquidate=False):
    result=(1+old*row['gap'])*(1-cost_bps/10000*abs(new-old))*(1+new*row['intraday'])-1
    if liquidate:result=(1+result)*(1-cost_bps/10000*abs(new))-1
    if not np.isfinite(result) or result<=-1:raise ValueError('Trading simulado sin capital o retorno no finito.')
    return float(result)


class TradingEnv:
    def __init__(self,X,meta,cost_bps=5,liquidate=True):
        self.X=np.asarray(X,dtype=np.float32);self.meta=meta;self.cost=cost_bps;self.liquidate=liquidate
    def reset(self):self.i=0;self.position=0.;return np.r_[self.X[0],self.position].astype(np.float32)
    def step(self,action):
        new=(-1.,0.,1.)[int(action)];done=self.i==len(self.X)-1;net=session_return(self.position,new,self.meta[self.i],self.cost,done and self.liquidate);self.position=new;self.i+=1
        state=np.r_[self.X[min(self.i,len(self.X)-1)],self.position].astype(np.float32)
        return state,100*net,done,{'netReturn':net,'position':new}


def run(datasets,cfg,retain_models=False,evaluate_test=True,algorithm='ddqn'):
    import tensorflow as tf
    if len(datasets)!=1:raise ValueError('Double DQN opera un activo por experimento para mantener una sola posición y curva de capital.')
    p=panel(datasets,cfg,1);s=scale_panel(p);X=s['X'][:,0,:];tr=p['masks']['train'];va=p['masks']['validation'];te=p['masks']['test'];seed=cfg['seed'];rng=np.random.default_rng(seed);tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
    episodes=bounded_int(cfg.get('episodes',10),1,100,'Episodios');capacity=bounded_int(cfg.get('replayCapacity',10000),128,100000,'Replay');batch=bounded_int(cfg.get('replayBatch',64),8,256,'Batch replay');warmup=bounded_int(cfg.get('replayWarmup',128),batch,capacity,'Calentamiento replay');every=bounded_int(cfg.get('trainEvery',4),1,64,'Pasos por actualización');sync=bounded_int(cfg.get('targetSync',100),1,10000,'Sincronización target');gamma=real(cfg.get('gamma',.95),0,.9999,'Gamma');epsstart=real(cfg.get('epsilonStart',1),0,1,'Epsilon inicial');epsend=real(cfg.get('epsilonEnd',.05),0,1,'Epsilon final');decay=bounded_int(cfg.get('epsilonDecay',5000),1,1000000,'Decaimiento epsilon');alpha=real(cfg.get('qAlpha',.15),.001,1,'Alpha Q');liquidate=bool(cfg.get('liquidateEnd',True))
    if epsend>epsstart:raise ValueError('Epsilon final debe ser menor o igual al inicial.')
    if episodes*int(tr.sum())>200000:raise ValueError('Presupuesto superior a 200,000 pasos de entrenamiento. Reduce episodios o registros.')
    def subset(mask):return X[mask],[r for r,b in zip(p['meta'],mask) if b]
    trainX,trainmeta=subset(tr);valX,valmeta=subset(va);testX,testmeta=subset(te)
    def network():
        a=cfg['architecture'];inp=tf.keras.Input((X.shape[1]+1,));h=tf.keras.layers.Dense(a['inputUnits'],activation=a['inputActivation'])(inp)
        for l in a['layers']:h=tf.keras.layers.Dense(l['units'],activation=l['activation'])(h)
        return tf.keras.Model(inp,tf.keras.layers.Dense(3,activation='linear')(h))
    online=network();target=network();target.set_weights(online.get_weights());opt=optimizer(tf,{**cfg['architecture'],'gradientClip':cfg['gradientClip'],'weightDecay':cfg['weightDecay']});memory=deque(maxlen=capacity);huber=tf.keras.losses.Huber();steps=0;updates=0;syncs=0;history=[];best=[-float('inf'),None,0,None];env=TradingEnv(trainX,trainmeta,cfg['costBps'],liquidate)
    def rollout(policy,features,meta):
        e=TradingEnv(features,meta,cfg['costBps'],liquidate);state=e.reset();positions=[]
        while True:
            action=int(policy(state));ns,_,done,info=e.step(action);positions.append(info['position']);state=ns
            if done:break
        return trade_path(meta,positions,cfg['costBps'],liquidate)
    for episode in range(episodes):
        state=env.reset();reward_sum=0.;lastloss=None
        while True:
            epsilon=epsend+(epsstart-epsend)*max(0.,1-steps/decay);action=int(rng.integers(3)) if rng.random()<epsilon else int(np.argmax(online(state[None],training=False).numpy()[0]));ns,reward,done,_=env.step(action);memory.append((state,action,reward,ns,done));state=ns;reward_sum+=reward;steps+=1
            if len(memory)>=warmup and steps%every==0:
                batchdata=[memory[int(i)] for i in rng.choice(len(memory),batch,replace=False)];ss,aa,rr,nn,dd=map(np.asarray,zip(*batchdata));expected=(double_targets(rr,dd,online(nn,training=False).numpy(),target(nn,training=False).numpy(),gamma) if algorithm=='ddqn' else rr+gamma*(1-dd)*target(nn,training=False).numpy().max(axis=1)).astype(np.float32)
                with tf.GradientTape() as tape:
                    q=online(ss,training=True);current=tf.gather(q,aa,axis=1,batch_dims=1);loss=huber(expected,current)
                grads=tape.gradient(loss,online.trainable_variables);opt.apply_gradients(zip(grads,online.trainable_variables));lastloss=float(loss);updates+=1
                if not np.isfinite(lastloss):raise ValueError('Double DQN divergente.')
                if updates%sync==0:target.set_weights(online.get_weights());syncs+=1
            if done:break
        validation=rollout(lambda state:int(np.argmax(online(state[None],training=False).numpy()[0])),valX,valmeta);score=validation['metrics']['netReturn'];history.append({'epoch':episode+1,'train':reward_sum,'validation':score,'epsilon':epsilon,'loss':lastloss})
        if score>best[0]:best[:]=[score,[w.copy() for w in online.get_weights()],episode+1,[w.copy() for w in target.get_weights()]]
    if updates==0:raise ValueError('No hubo actualizaciones de Double DQN. Reduce warmup/lote o aumenta episodios.')
    online.set_weights(best[1]);target.set_weights(best[3]);policy=lambda state:int(np.argmax(online(state[None],training=False).numpy()[0]));valresult=rollout(policy,valX,valmeta)
    if not evaluate_test:return {'comparison':[{'model':algorithm.upper(),'validation':valresult['metrics'],'history':history,'parameters':online.count_params(),'bestEpoch':best[2]}],'config':cfg,'_online':online,'_target':target,'_panel':p,'_scale':s,'_testX':testX,'_testMeta':testmeta}
    testresult=rollout(policy,testX,testmeta)
    # Tabular baseline shares environment and training budget; validation selects its checkpoint.
    Q={};tablebest=[-float('inf'),None];table_rng=np.random.default_rng(seed);table_steps=0
    vol_index=cfg['selectedFeatures'].index('volatility20') if 'volatility20' in cfg['selectedFeatures'] else 0
    def state_key(state):return (int(np.digitize(state[0],[-1,-.25,.25,1])),int(np.digitize(state[vol_index],[-1,0,1])),int(state[-1]))
    def qvalues(state):return Q.setdefault(state_key(state),np.zeros(3))
    for episode in range(episodes):
        state=env.reset()
        while True:
            epsilon=epsend+(epsstart-epsend)*max(0.,1-table_steps/decay);action=int(table_rng.integers(3)) if table_rng.random()<epsilon else int(qvalues(state).argmax());ns,r,done,_=env.step(action);qvalues(state)[action]+=alpha*(r+(0 if done else gamma*qvalues(ns).max())-qvalues(state)[action]);state=ns;table_steps+=1
            if done:break
        value=rollout(lambda state:int(Q.get(state_key(state),np.zeros(3)).argmax()),valX,valmeta)['metrics']['netReturn']
        if value>tablebest[0]:tablebest=[value,{key:val.copy() for key,val in Q.items()}]
    Q=tablebest[1];tablepolicy=lambda state:int(Q.get(state_key(state),np.zeros(3)).argmax())
    candidates=[('Double DQN',valresult,testresult),('Q-Learning tabular',rollout(tablepolicy,valX,valmeta),rollout(tablepolicy,testX,testmeta)),('Siempre largo',trade_path(valmeta,np.ones(len(valmeta)),cfg['costBps'],liquidate),trade_path(testmeta,np.ones(len(testmeta)),cfg['costBps'],liquidate)),('Efectivo',trade_path(valmeta,np.zeros(len(valmeta)),cfg['costBps'],liquidate),trade_path(testmeta,np.zeros(len(testmeta)),cfg['costBps'],liquidate))]
    comparison=[{'model':name,'validation':v['metrics'],'test':t['metrics'],'parameters':online.count_params() if name=='Double DQN' else None,'bestEpoch':best[2] if name=='Double DQN' else None,'history':history if name=='Double DQN' else []} for name,v,t in candidates];comparison.sort(key=lambda r:-r['validation']['netReturn'])
    return {'unit':10,'task':'trading','winner':'Double DQN','bestValidationStrategy':comparison[0]['model'],'criterion':'checkpoint Double DQN por retorno neto de validación','config':{**cfg,'episodes':episodes,'replayCapacity':capacity,'replayWarmup':warmup,'replayBatch':batch,'trainEvery':every,'targetSync':sync,'gamma':gamma,'epsilonStart':epsstart,'epsilonEnd':epsend,'epsilonDecay':decay,'qAlpha':alpha,'liquidateEnd':liquidate},'features':cfg['selectedFeatures']+['posición actual'],'splits':split_summary(p),'purgedSamples':p['purged'],'excludedSamples':p['excluded'],'comparison':comparison,'test':{'metrics':testresult['metrics'],'path':testresult['path']},'testStrategies':[{'model':name,**t} for name,v,t in candidates],'training':{'steps':steps,'updates':updates,'targetSynchronizations':syncs,'replaySize':len(memory),'replayContains':'solo train','bestEpisode':best[2]},'network':{'inputShape':[X.shape[1]+1],'outputShape':[3],'parameters':online.count_params()},'note':'Decisión al cierre t: posición anterior participa en gap t→t+1; nueva posición entra en apertura t+1. Coste por unidad de cambio de posición y liquidación final configurable. Double DQN: online elige, target evalúa; Huber, clipping, replay solo train. Validación/test greedy sin aprendizaje. Sharpe anualizado con 252 sesiones; simulación diaria sin margen ni financiación.'}
