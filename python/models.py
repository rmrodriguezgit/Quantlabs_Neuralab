"""Modelos ajustados exclusivamente con la ventana de entrenamiento recibida."""
import numpy as np

METHODS = ['naive','linear','polynomial','arima','sarima','garch','monte_carlo','rnn','cnn','lstm','prophet','qlearning','dqn','gan']

def predict(name, values, steps, period=12, seed=42, options=None):
    y=np.asarray(values,dtype=float)
    options=options or {}
    rng=np.random.default_rng(seed)
    extra={}
    if len(y)<12: raise ValueError('Se requieren al menos 12 observaciones de entrenamiento.')
    if name=='naive': pred=np.repeat(y[-1],steps)
    elif name in ('linear','polynomial'):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LinearRegression
        model=make_pipeline(StandardScaler(),PolynomialFeatures(1 if name=='linear' else 2),LinearRegression())
        model.fit(np.arange(len(y)).reshape(-1,1),y)
        pred=model.predict(np.arange(len(y),len(y)+steps).reshape(-1,1))
    elif name in ('arima','sarima'):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        if name=='sarima' and (period<2 or len(y)<3*period): raise ValueError('SARIMA requiere al menos tres ciclos y periodo ≥2.')
        center,scale=float(y.mean()),max(float(y.std()),1e-8)
        specification=SARIMAX((y-center)/scale,order=(1,1,1),seasonal_order=(1,0,0,period) if name=='sarima' else (0,0,0,0),enforce_stationarity=False,enforce_invertibility=False)
        model=specification.fit(disp=False,maxiter=200)
        if not model.mle_retvals.get('converged',True):model=specification.fit(method='powell',disp=False,maxiter=300)
        if not model.mle_retvals.get('converged',True): raise ValueError('El ajuste ARIMA/SARIMA no convergió.')
        result=model.get_forecast(steps)
        pred=np.asarray(result.predicted_mean)*scale+center
        ci=np.asarray(result.conf_int(alpha=.05))*scale+center
        extra={'lower':ci[:,0].tolist(),'upper':ci[:,1].tolist(),'intervalLabel':'Intervalo del modelo al 95%'}
    elif name in ('garch','monte_carlo'):
        if np.any(y<=0): raise ValueError('Este modelo requiere valores estrictamente positivos.')
        returns=np.diff(np.log(y))
        if name=='garch':
            from arch import arch_model
            model=arch_model(returns*100,mean='Constant',vol='GARCH',p=1,q=1,rescale=False).fit(disp='off')
            if model.convergence_flag!=0: raise ValueError('GARCH no convergió.')
            f=model.forecast(horizon=steps,reindex=False)
            mu=np.asarray(f.mean)[-1]/100
            sigma=np.sqrt(np.asarray(f.variance)[-1])/100
            extra['volatility']=sigma.tolist()
            shocks=rng.normal(mu,sigma,size=(1000,steps))
        else:
            shocks=rng.normal(returns.mean(),returns.std(ddof=1),size=(1000,steps))
        paths=y[-1]*np.exp(np.cumsum(shocks,axis=1))
        pred=np.median(paths,axis=0)
        extra.update(lower=np.quantile(paths,.025,axis=0).tolist(),upper=np.quantile(paths,.975,axis=0).tolist(),intervalLabel='Percentiles simulados 2.5–97.5%; no calibrados')
    elif name in ('rnn','cnn','lstm'):
        from neural import neural_forecast
        return neural_forecast(name,y,steps,seed,options)
    elif name=='gan':
        from neural import gan_forecast
        return gan_forecast(y,steps,seed,options)
    elif name=='dqn':
        from neural import dqn_forecast
        return dqn_forecast(y,steps,seed,options)
    elif name=='prophet':
        import pandas as pd
        from prophet import Prophet
        # Synthetic daily index preserves observation spacing; seasonal periods are in observations.
        frame=pd.DataFrame({'ds':pd.date_range('2000-01-01',periods=len(y),freq='D'),'y':y})
        model=Prophet(daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False)
        if period>=2 and len(y)>=2*period: model.add_seasonality(name='cycle',period=period,fourier_order=min(5,max(1,period//2)))
        model.fit(frame)
        f=model.predict(pd.DataFrame({'ds':pd.date_range(frame.ds.iloc[-1]+pd.Timedelta(days=1),periods=steps,freq='D')}))
        pred=f.yhat.to_numpy();extra={'lower':f.yhat_lower.tolist(),'upper':f.yhat_upper.tolist(),'intervalLabel':'Intervalo Prophet al 80% (predeterminado)'}
    elif name=='qlearning':
        return _reinforcement(name,y,steps,seed,options)
    else: raise ValueError('Método desconocido: '+name)
    pred=np.asarray(pred,dtype=float)
    if len(pred)!=steps or not np.isfinite(pred).all(): raise ValueError('Pronóstico no finito o longitud inválida.')
    return {'prediction':pred.tolist(),**extra}

def _reinforcement(name,y,steps,seed,options):
    """Política experimental de dirección: estados discretos, sin operar dinero."""
    import gymnasium as gym
    from gymnasium import spaces
    rng=np.random.default_rng(seed)
    scale=max(float(np.std(np.diff(y))),1e-8)
    changes=np.diff(y)/scale
    edges=np.array([-1.,-.2,.2,1.])
    state=lambda x:int(np.digitize(x,edges))
    class DirectionEnv(gym.Env):
        observation_space=spaces.Discrete(5)
        action_space=spaces.Discrete(3)
        def reset(self,seed=None,options=None):
            super().reset(seed=seed);self.i=0;return state(changes[0]),{}
        def step(self,action):
            self.i+=1
            reward=-float((changes[self.i]-(action-1))**2)
            return state(changes[self.i]),reward,self.i>=len(changes)-1,False,{}
    env=DirectionEnv()
    episodes=max(10,min(500,int(options.get('episodes',60))))
    alpha=float(options.get('alpha',.15));gamma=float(options.get('gamma',.9));epsilon=float(options.get('epsilon',.15))
    if not (0<alpha<=1 and 0<=gamma<1 and 0<=epsilon<=1): raise ValueError('Parámetros RL fuera de rango.')
    rewards=[]
    Q=np.zeros((5,3))
    for episode in range(episodes):
        s,_=env.reset(seed=seed+episode);total=0
        while True:
            a=int(rng.integers(3)) if rng.random()<epsilon else int(Q[s].argmax())
            ns,r,done,_,_=env.step(a)
            Q[s,a]+=alpha*(r+(0 if done else gamma*Q[ns].max())-Q[s,a]);s=ns;total+=r
            if done:break
        rewards.append(total)
    policy=lambda s:int(Q[s].argmax())
    out=[];last=float(y[-1]);s=state(changes[-1]);actions=[]
    for _ in range(steps):
        a=policy(s);last+=(a-1)*scale;out.append(last);actions.append(a-1);s=state(a-1)
    return {'prediction':out,'actions':actions,'trainingRewards':rewards,'note':'Política experimental de dirección; sin rentabilidad ni validación de estrategia de trading.'}
