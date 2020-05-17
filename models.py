from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd
import statsmodels.formula.api as sm
from pydantic import BaseModel
from statsmodels.regression.linear_model import RegressionResultsWrapper

from data_creation import Parameters


class ModelResults(BaseModel):
    lower_bounds: Parameters
    mean: Parameters
    upper_bounds: Parameters


class Model(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> ModelResults:
        pass


class OLSModel(Model):
    def fit(self, data: pd.DataFrame) -> ModelResults:
        try:
            self.formula
        except AttributeError:
            print("Does not have a formula")
        ols = sm.ols(formula=self.formula, data=data).fit()
        return self.get_results(ols.params, ols.conf_int())

    def get_results(
        self, params: Dict[str, float], conf_int: pd.DataFrame
    ) -> ModelResults:
        def get_parameters(res: Dict[str, float]) -> Parameters:
            intercept = 0.0
            parameter_of_interest = 0.0
            school_intercepts = []
            air_quality = 0.0
            school_air_quality = []
            for parameter, value in res.items():
                if parameter == "Intercept":
                    intercept = value
                elif parameter == "hours_studying":
                    parameter_of_interest = value
                elif parameter == "air_quality":
                    air_quality = value
                elif ("school" in parameter) & ("air_quality" in parameter):
                    school_air_quality.append(value)
                elif "school" in parameter:
                    school_intercepts.append(value)
                else:
                    continue
            return Parameters(
                school_intercepts=school_intercepts,
                school_air_quality=school_air_quality,
                air_quality=air_quality,
                parameter_of_interest=parameter_of_interest,
                global_intercept=intercept,
            )

        return ModelResults(
            mean=get_parameters(params),
            lower_bounds=get_parameters(conf_int[0]),
            upper_bounds=get_parameters(conf_int[1]),
        )


# make and ols class
# makes the formula the name and have it in the init
# have a get results that checks formula for values
# if it has state in it then get state_intercepts
# if it state_vals then get state slopes


class NormalOLS(OLSModel):
    def __init__(self) -> None:
        self.formula = "test_scores ~ hours_studying"


class FEInterceptOLS(OLSModel):
    def __init__(self) -> None:
        self.formula = "test_scores ~ hours_studying + C(school)"


class FullFEOLS(OLSModel):
    def __init__(self) -> None:
        self.formula = (
            "test_scores ~ hours_studying + C(school) * air_quality + C(school)"
        )


class REInterceptOLS(OLSModel):
    def __init__(self) -> None:
        self.formula = "test_scores ~ hours_studying"

    def fit(self, data: pd.DataFrame) -> ModelResults:
        try:
            self.formula
        except AttributeError:
            print("Does not have a formula")
        results = sm.mixedlm(
            formula=self.formula, data=data, groups=data["school"],
        ).fit()
        params = results.params.to_dict()
        for re, value in results.random_effects.items():
            params[f"school_{re}"] = value["Group"]

        conf_int = results.conf_int()
        new_conf_int = {}
        for i in [0, 1]:
            conf_int_vals = conf_int[i].to_dict()
            for re, value in results.random_effects.items():
                if i == 0:
                    conf_int_vals[f"school_{re}"] = value["Group"] - (
                        params["Group Var"] * 1.96
                    )
                if i == 1:
                    conf_int_vals[f"school_{re}"] = value["Group"] + (
                        params["Group Var"] * 1.96
                    )
            new_conf_int[i] = conf_int_vals

        return self.get_results(params, pd.DataFrame(new_conf_int))


class REFullOLS(OLSModel):
    def __init__(self) -> None:
        self.formula = "test_scores ~ hours_studying"

    def fit(self, data: pd.DataFrame) -> ModelResults:
        try:
            self.formula
        except AttributeError:
            print("Does not have a formula")
        results = sm.mixedlm(
            formula=self.formula,
            data=data,
            groups=data["school"],
            re_formula="~air_quality",
        ).fit()

        params = results.params.to_dict()
        for re, value in results.random_effects.items():
            params[f"school_{re}"] = value["Group"]
            params[f"school_{re}:air_quality"] = value["air_quality"]

        conf_int = results.conf_int()
        new_conf_int = {}
        for i in [0, 1]:
            conf_int_vals = conf_int[i].to_dict()
            for re, value in results.random_effects.items():
                if i == 0:
                    conf_int_vals[f"school_{re}"] = value["Group"] - (
                        params["Group Var"] * 1.96
                    )
                    conf_int_vals[f"school_{re}:air_quality"] = value["air_quality"] - (
                        params["air_quality Var"] * 1.96
                    )
                if i == 1:
                    conf_int_vals[f"school_{re}"] = value["Group"] + (
                        params["Group Var"] * 1.96
                    )
                    conf_int_vals[f"school_{re}:air_quality"] = value["air_quality"] + (
                        params["air_quality Var"] * 1.96
                    )
            new_conf_int[i] = conf_int_vals

        return self.get_results(params, pd.DataFrame(new_conf_int))
