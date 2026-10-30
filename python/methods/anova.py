import numpy as np
import pandas as pd
from domain.contracts import MethodSpec,VariantSpec
from .common import output,params
SPECS=(VariantSpec(1,'ANOVA de un factor','Fase estacional como factor y pronóstico de medias por fase.'),VariantSpec(2,'Dos factores e interacción','Fase × tramo temporal histórico; futuro mantiene el último tramo conocido.'),VariantSpec(3,'ANCOVA y MANOVA opcional','Covariables rezagadas para la respuesta seleccionada; MANOVA si existen dos respuestas.',{'maxCovariates':3}))
def forecast(ctx,variant):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    p=params(ctx,variant,SPECS);n=len(ctx.values);period=ctx.period;t=np.arange(n);phase=t%max(1,period);era=np.minimum(2,t*3//n);data=pd.DataFrame({'y':ctx.values,'phase':phase,'era':era,'t':t/max(n-1,1)});future=pd.DataFrame({'phase':(n+np.arange(ctx.horizon))%max(1,period),'era':np.repeat(era[-1],ctx.horizon),'t':(n+np.arange(ctx.horizon))/max(n-1,1)})
    formula='y ~ C(phase)' if variant==1 else 'y ~ C(phase)*C(era)' if variant==2 else 'y ~ C(phase) + t';covs=[];manova=None
    if variant==3:
        from features import bounded_int
        columns=[c for c in ctx.frame.columns if c!=ctx.column and np.isfinite(ctx.frame[c].to_numpy(dtype=float)).all()][:bounded_int(p.get('maxCovariates',3),0,5,'Covariables')]
        for i,col in enumerate(columns):
            key='x'+str(i);data[key]=ctx.frame[col].astype(float).shift(1).to_numpy();future[key]=float(ctx.frame[col].iloc[-1]);formula+=' + '+key;covs.append(col)
        if columns:
            try:
                from statsmodels.multivariate.manova import MANOVA
                responses=np.c_[ctx.values,ctx.frame[columns[0]].to_numpy(dtype=float)];exog=sm.add_constant(np.c_[data.t.to_numpy(),np.sin(2*np.pi*t/max(1,period)),np.cos(2*np.pi*t/max(1,period))]);manova=str(MANOVA(responses,exog).mv_test())
            except (ValueError,np.linalg.LinAlgError) as e:manova='MANOVA no estimable: '+str(e)
        else:manova='MANOVA no aplicable: requiere al menos dos respuestas. Se ejecuta ANCOVA univariada con tiempo como covariable.'
    data=data.dropna();m=smf.ols(formula,data=data).fit()
    if m.df_resid<=0:raise ValueError('ANOVA sin grados de libertad residuales: amplía historial o reduce periodo/covariables.')
    try:
        table=sm.stats.anova_lm(m,typ=2);anova=[{'term':str(k),**{str(c):float(v) for c,v in row.items()}} for k,row in table.iterrows()]
    except (ValueError,np.linalg.LinAlgError) as e:anova=[{'note':str(e)}]
    return output(m.predict(future).to_numpy(),ctx,formula=formula,anova=anova,manova=manova,covariates=covs,futureCovariates='Último valor observado en TR; escenario constante, no valores reales de VA/TE.',note='ANOVA/ANCOVA estudian asociaciones, no prueban causalidad. Autocorrelación puede invalidar p-valores clásicos. MANOVA es descriptiva multirrespuesta y no sustituye el pronosticador ANCOVA.')
STUDY=MethodSpec('anova','ANOVA / ANCOVA / MANOVA','anova',SPECS,forecast,('Definir respuesta y factores antes de inspeccionar test.','Estimar efectos e interacción solo en entrenamiento.','Examinar grados de libertad y residuos.','Distinguir ANCOVA de MANOVA; revisar número de respuestas.','Pronosticar bajo escenarios explícitos de covariables futuras.'))
