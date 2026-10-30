from dataclasses import dataclass,field
from typing import Callable
import numpy as np
import pandas as pd
@dataclass(frozen=True)
class StudyContext:
    values:np.ndarray
    index:pd.DatetimeIndex
    frame:pd.DataFrame
    column:str
    horizon:int
    period:int
    seed:int
    settings:dict=field(default_factory=dict)
    parameters:dict=field(default_factory=dict)
    frequency:str='business'
    def prefix(self,end,horizon=None):
        return StudyContext(self.values[:end].copy(),self.index[:end].copy(),self.frame.iloc[:end].copy(),self.column,horizon or self.horizon,self.period,self.seed,dict(self.settings),dict(self.parameters),self.frequency)
@dataclass(frozen=True)
class VariantSpec:
    number:int
    title:str
    description:str
    defaults:dict=field(default_factory=dict)
@dataclass(frozen=True)
class MethodSpec:
    key:str
    title:str
    lesson:str
    variants:tuple
    forecast:Callable
    theory:tuple
    dependency:str|None=None
