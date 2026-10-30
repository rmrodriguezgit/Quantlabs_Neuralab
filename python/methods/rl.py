import numpy as np
from domain.contracts import MethodSpec,VariantSpec
from domain.decomposition import causal
from scaling import fit_affine
from .common import output,params,bounded_int,real
SPECS=(VariantSpec(1,'Q-Learning tabular','Estados discretos de cambio/tendencia y pérdida cuadrática.',{'episodes':10,'alpha':.15,'gamma':.95,'epsilon':.2}),VariantSpec(2,'Deep Q-Network','Estado continuo, replay de TR y red target.',{'episodes':10,'units':[32,16],'gamma':.95,'epsilon':.2,'batchSize':64,'trainEvery':16}),VariantSpec(3,'Double DQN · mejora MSE','Online selecciona y target evalúa; recompensa = reducción del error frente a no cambio.',{'episodes':10,'units':[32,16],'gamma':.95,'epsilon':.2,'batchSize':64,'trainEvery':16}))
def td_targets(online,target,rewards,done,gamma,double):
    future=target[np.arange(len(target)),np.argmax(online,axis=1)] if double else target.max(axis=1)
    return rewards+gamma*(1-done)*future

def features(values,period,scale):
    y=np.asarray(values);c=causal(y,period);return np.c_[np.r_[0,np.diff(y)]/scale,np.r_[0,np.diff(c[:,0])]/scale,c[:,1]/scale,c[:,2]/scale]

def forecast(ctx,variant):
    p=params(ctx,variant,SPECS);y=ctx.values;scale=max(float(np.std(np.diff(y))),1e-8);raw=features(y,ctx.period,scale);xc,xs=fit_affine(raw[:-1]);states=((raw-xc)/xs).astype(np.float32);ss=states[1:-1];nn=states[2:];delta=np.diff(y)[1:]/scale;n=len(ss)
    if n<16:raise ValueError('RL requiere al menos dieciocho observaciones.')
    episodes=bounded_int(p.get('episodes',10),1,100,'Episodios');gamma=real(p.get('gamma',.95),0,.9999,'Gamma');epsilon=real(p.get('epsilon',.2),0,1,'Epsilon');rng=np.random.default_rng(ctx.seed);history=[]
    if n*episodes>200000:raise ValueError('RL supera 200,000 transiciones; reduce historial o episodios.')
    def bucket(x):return int(np.digitize(x[0],[-1,-.2,.2,1]))+5*int(np.digitize(x[1],[-.2,.2]))
    if variant==1:
        q=np.zeros((15,3));alpha=real(p.get('alpha',.15),.0001,1,'Alpha')
        for ep in range(episodes):
            total=0.
            for i in range(n):
                s=bucket(ss[i]);a=int(rng.integers(3)) if rng.random()<epsilon else int(q[s].argmax());r=-(delta[i]-(a-1))**2;future=0 if i==n-1 else gamma*q[bucket(nn[i])].max();q[s,a]+=alpha*(r+future-q[s,a]);total+=r
            history.append({'episode':ep+1,'reward':float(total)})
        choose=lambda state:int(q[bucket(state)].argmax())-1;layers=[]
    else:
        import tensorflow as tf
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(ctx.seed);units=p.get('units',[32,16])
        if not isinstance(units,list) or not 1<=len(units)<=5:raise ValueError('RL: una a cinco capas.')
        units=[bounded_int(v,1,128,'Unidades RL') for v in units]
        def net():return tf.keras.Sequential([tf.keras.Input((4,)),*[tf.keras.layers.Dense(u,activation='relu') for u in units],tf.keras.layers.Dense(3,activation='linear')])
        online=net();target=net();target.set_weights(online.get_weights());online.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=real(p.get('learningRate',.001),1e-6,.1,'LR'),global_clipnorm=1),loss=tf.keras.losses.Huber());batch=bounded_int(p.get('batchSize',64),8,256,'Batch RL');every=bounded_int(p.get('trainEvery',16),1,128,'Frecuencia RL');sync=bounded_int(p.get('targetSync',50),1,10000,'Sincronización');updates=0;buffer=[]
        for ep in range(episodes):
            values=online(ss,training=False).numpy();actions=values.argmax(axis=1);explore=rng.random(n)<epsilon;actions[explore]=rng.integers(3,size=int(explore.sum()));rewards=-(delta-(actions-1))**2
            if variant==3:rewards=delta**2+rewards
            buffer.extend((ss[i],actions[i],rewards[i],nn[i],float(i==n-1)) for i in range(n));buffer=buffer[-10000:];loss=None
            if len(buffer)>=batch:
                for _ in range(min(200,max(1,n//every))):
                    chosen=rng.choice(len(buffer),batch,replace=False);x,a,r,nx,d=map(np.asarray,zip(*[buffer[i] for i in chosen]));q=online(x,training=False).numpy();expected=td_targets(online(nx,training=False).numpy(),target(nx,training=False).numpy(),r,d,gamma,variant==3);q[np.arange(batch),a]=expected;loss=float(online.train_on_batch(x,q));updates+=1
                    if not np.isfinite(loss):raise ValueError('RL: pérdida no finita; reduce LR o revisa datos.')
                    if updates%sync==0:target.set_weights(online.get_weights())
            history.append({'episode':ep+1,'reward':float(rewards.sum()),'loss':loss})
        choose=lambda state:int(online(state[None],training=False).numpy()[0].argmax())-1;layers=[{'name':l.name,'type':type(l).__name__} for l in online.layers]
    values=list(y);prediction=[];actions=[]
    for _ in range(ctx.horizon):
        state=((features(values,ctx.period,scale)[-1]-xc)/xs).astype(np.float32);action=choose(state);next_value=values[-1]+action*scale;values.append(next_value);prediction.append(next_value);actions.append(action)
    return output(prediction,ctx,actions=actions,history=history,layers=layers,episodes=episodes,replayContains='solo TR' if variant>1 else 'no aplica',reward='delta² − (delta − acción)²' if variant==3 else '−(delta − acción)²',note='Estudio de dirección/reducción de error, no simulación de rentabilidad. Estado de mercado observado no depende de la acción; replay y estadísticas pertenecen solo a TR. Usa episodios, no ES por épocas.')
STUDY=MethodSpec('rl','Q-Learning / DQN / Double DQN','ddqn',SPECS,forecast,('Definir estados conocidos y tres acciones de cambio.','Normalizar cambios solo con TR.','Aprender tabularmente o con replay y target.','En DDQN separar elección y evaluación; enmascarar terminales.','Evaluar error de pronóstico; no inferir rentabilidad de la recompensa.'),'tensorflow')
