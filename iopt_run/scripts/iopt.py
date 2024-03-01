import argparse
import logging

import pandas as pd

from statistics import mean
from examples.Genetic_algorithm.TSP._2D.Example_GA_TSP_Vary_Mutation_Population_Size import (
    load_TSPs_matrix,
)
from examples.Genetic_algorithm.TSP._2D.Problems.ga_tsp_2d import GA_TSP_2D
from examples.Machine_learning.SVC._2D.Example_2D_SVC import load_breast_cancer_data
from examples.Machine_learning.SVC._2D.Problems.SVC_2d import SVC_2D
from iOpt.solver import Solver
from iOpt.solver_parametrs import SolverParameters
from iOpt.output_system.listeners.console_outputers import ConsoleOutputListener
from problems.GKLS import GKLS


URL_DB = "postgresql://10.0.2.20:5432/iopt_db"
r = 4


class IOptExperiment:
    def __init__(self, args):
        self.args = args
        self.time = 0
        self.value = 0

        if self.args.task == "GKLS":
            self.eps = 0.01
            self.n_iter = 20000
            self.problems = [GKLS(dimension=3, functionNumber=i) for i in range(1, 101)]
        elif self.args.task == "SVC_2D":
            x, y = load_breast_cancer_data()
            regularization_value_bound = {"low": 1, "up": 6}
            kernel_coefficient_bound = {"low": -7, "up": -3}
            self.eps = 1e-12
            self.n_iter = 1000
            self.problems = [
                SVC_2D(x, y, regularization_value_bound, kernel_coefficient_bound)
            ] * self.args.launches
        elif self.args.task == "GA_TSP_2D":
            tsp_matrix = load_TSPs_matrix(
                "/common/home/zorin_o/iOpt/examples/Genetic_algorithm/TSP/TSPs_matrices/a280.xml"
            )
            num_iteration = 200
            mutation_probability_bound = {"low": 0.0, "up": 1.0}
            population_size_bound = {"low": 10.0, "up": 100.0}
            self.eps = 1e-12
            self.n_iter = 200
            self.problems = [
                GA_TSP_2D(
                    tsp_matrix,
                    num_iteration,
                    mutation_probability_bound,
                    population_size_bound,
                )
            ] * self.args.launches

    def calculate_data(self):
        all_time = []
        all_value = []
        params = SolverParameters(
            r=r,
            eps=self.eps,
            iters_limit=self.n_iter,
            number_of_parallel_points=self.args.number_of_parallel_points,
            async_scheme=self.args.is_async,
            url_db=URL_DB if self.args.db else None,
        )
        for i, problem in enumerate(self.problems):
            logging.info(
                f"start {self.args.task} {i} (n={self.args.number_of_parallel_points}, async={self.args.is_async}, db={self.args.db})"
            )

            solver = Solver(problem, params)
            # cfol = ConsoleOutputListener(mode="full")
            # solver.add_listener(cfol)
            sol_info = solver.solve()

            all_time.append(sol_info.solving_time)
            all_value.append(sol_info.best_trials[0].function_values[0].value)

            logging.info(f"solved {self.args.task} {i}")
        self.time = mean(all_time)
        self.value = mean(all_value)

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
                "GKLS": "GKLS\nЧисло испытаний 1000",
                "GA_TSP_2D": f"GA_TSP_2D\nЧисло испытаний 200\n{self.args.launches} запусков",
                "SVC_2D": f"SVC_2D\nЧисло испытаний 1000\n{self.args.launches} запусков",
            },
            index={0: self.args.number_of_parallel_points},
        )
        name = self.args.task + f"_n{self.args.number_of_parallel_points}"
        if self.args.is_async:
            name += "_async"
        if self.args.db:
            name += "_db"
        df.to_excel(f"{name}.xlsx", float_format="%.3f")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--task", type=str, choices=["GKLS", "GA_TSP_2D", "SVC_2D"], required=True
    )
    parser.add_argument("-l", "--launches", type=int, default=1)
    parser.add_argument("-n", "--number_of_parallel_points", type=int, default=1)
    parser.add_argument("-a", "--is_async", type=int, choices=[0, 1], default=0)
    parser.add_argument("-d", "--db", type=int, choices=[0, 1], default=0)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
    arguments = parse_arguments()
    ex = IOptExperiment(arguments)
    ex.calculate_data()
    ex.save_data()
