import logging
from time import perf_counter

import pandas as pd
import optuna
import argparse
import numpy as np

from examples.Genetic_algorithm.TSP._2D.Example_GA_TSP_Vary_Mutation_Population_Size import (
    load_TSPs_matrix,
)
from examples.Genetic_algorithm.TSP._2D.Problems.ga_tsp_2d import GA_TSP_2D
from examples.Machine_learning.SVC._2D.Example_2D_SVC import load_breast_cancer_data
from examples.Machine_learning.SVC._2D.Problems.SVC_2d import SVC_2D
from iOpt.problem import Problem
from iOpt.trial import Point, FunctionValue
from problems.GKLS import GKLS


def create_objective(problem: Problem):
    def objective(trial):
        list_vars = [
            trial.suggest_float(f"x{i}", low, up)
            for i, (low, up) in enumerate(
                zip(
                    problem.lower_bound_of_float_variables,
                    problem.upper_bound_of_float_variables,
                )
            )
        ]
        return problem.calculate(Point(list_vars), FunctionValue()).value

    return objective


class OptunaExperiment:
    def __init__(self, args):
        self.args = args

        if self.args.task == "GKLS":
            self.n_iter = 1000
            self.objectives = [
                create_objective(GKLS(dimension=3, functionNumber=i))
                for i in range(1, 101)
            ]
        elif self.args.task == "SVC_2D":
            x, y = load_breast_cancer_data()
            regularization_value_bound = {"low": 1, "up": 6}
            kernel_coefficient_bound = {"low": -7, "up": -3}
            problem = SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)
            objective = create_objective(problem)
            self.n_iter = 1000
            self.objectives = [objective] * self.args.launches
        elif self.args.task == "GA_TSP_2D":
            tsp_matrix = load_TSPs_matrix(
                "/common/home/zorin_o/iOpt/examples/Genetic_algorithm/TSP/TSPs_matrices/a280.xml"
            )
            num_iteration = 200
            mutation_probability_bound = {"low": 0.0, "up": 1.0}
            population_size_bound = {"low": 10.0, "up": 100.0}
            problem = GA_TSP_2D(
                tsp_matrix,
                num_iteration,
                mutation_probability_bound,
                population_size_bound,
            )
            objective = create_objective(problem)
            self.n_iter = 200
            self.objectives = [objective] * self.args.launches

    def calculate_data(self):
        all_time = []
        all_values = []
        for i, objective in enumerate(self.objectives):
            logging.info(f"start {self.args.task} {i}")

            study = optuna.create_study()
            start_time = perf_counter()
            study.optimize(objective, self.n_iter, n_jobs=self.args.n_jobs)
            all_time.append(perf_counter() - start_time)
            all_values.append(study.best_value)

            logging.info(f"solved {self.args.task} {i}")
        self.time = np.mean(all_time)
        self.value = np.mean(all_values)

    def save_data(self):
        data = {
            (self.args.task, "time"): [self.time],
            (self.args.task, "value"): [self.value],
        }
        df = pd.DataFrame(data)
        df = df.rename_axis("Число процессов")
        df = df.rename(
            columns={
                "time": "Среднее время вычисления (сек.)",
                "value": "Средний результат",
                "GKLS": "GKLS (с задержкой в функции 0.01 сек.)\nЧисло испытаний 1000",
                "GA_TSP_2D": f"GA_TSP_2D\nЧисло испытаний 200\n{self.args.launches} запусков",
                "SVC_2D": f"SVC_2D\nЧисло испытаний 1000\n{self.args.launches} запусков",
            },
            index={0: self.args.n_jobs},
        )
        df.to_excel(f"{self.args.task}_n{self.args.n_jobs}_thread.xlsx", float_format="%.3f")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_jobs", type=int, default=1)
    parser.add_argument("-l", "--launches", type=int, default=1)
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["GKLS", "GA_TSP_2D", "SVC_2D"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s %(processName)s %(process)s %(message)s",
        level=logging.INFO,
    )
    arguments = parse_arguments()
    t = OptunaExperiment(arguments)
    t.calculate_data()
    t.save_data()
