from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel


class Parameters(BaseModel):
    school_intercepts: Optional[List[float]]
    school_air_quality: Optional[List[float]]
    air_quality: float
    global_intercept: float
    parameter_of_interest: float


class Noise(BaseModel):
    individual_noise: int
    school_noise: int


class DataGeneration:
    def __init__(self, num_schools: int):
        self.num_schools = num_schools

    def simulate_data(
        self, data_points: int, noise: Noise
    ) -> Tuple[pd.DataFrame, Parameters]:
        """
        data points is the number of data points in our data
        """

        # might want to change all of the samples to be from normal
        # can look at how changing this affects the outcome

        school_intercepts = np.random.normal(10, 1, self.num_schools)
        school_slopes = np.random.normal(10, 1, self.num_schools)

        # eg average income of school
        school_air_quality = np.random.normal(10, 1, data_points)
        school_values = np.random.randint(0, self.num_schools, data_points)

        school_impact = (
            school_intercepts[school_values]
            + school_slopes[school_values] * school_air_quality
            + np.random.normal(0, noise.school_noise, data_points)
        )

        global_intercept = np.random.randint(0, 10)

        parameter_of_interest = np.random.randint(0, 10)
        value_of_interest = np.random.normal(10, 1, data_points)

        y_vals = (
            global_intercept + parameter_of_interest * value_of_interest + school_impact
        ) + np.random.normal(0, noise.individual_noise, data_points)

        data = pd.DataFrame(
            {
                "test_scores": y_vals,
                "hours_studying": value_of_interest,
                "school": school_values,
                "air_quality": school_air_quality,
            }
        )
        true_parameters = Parameters(
            school_intercepts=school_intercepts.tolist(),
            school_air_quality=(school_slopes - school_slopes.mean()).tolist(),
            air_quality=school_slopes.mean(),
            global_intercept=global_intercept,
            parameter_of_interest=parameter_of_interest,
        )
        return data, true_parameters
