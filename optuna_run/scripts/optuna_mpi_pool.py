import logging
from time import perf_counter

import pandas as pd
import optuna
import argparse
import numpy as np
from mpi4py import MPI
from pathos.multiprocessing import ProcessingPool

from examples.Genetic_algorithm.TSP._2D.Example_GA_TSP_Vary_Mutation_Population_Size import (
    load_TSPs_matrix,
)
from examples.Genetic_algorithm.TSP._2D.Problems.ga_tsp_2d import GA_TSP_2D
from examples.Machine_learning.SVC._2D.Example_2D_SVC import load_breast_cancer_data
from examples.Machine_learning.SVC._2D.Problems.SVC_2d import SVC_2D
from iOpt.problem import Problem
from iOpt.trial import Point, FunctionValue
from problems.GKLS import GKLS

# optuna.logging.set_verbosity(optuna.logging.WARN)

URL_DB = "postgresql://mynonsuperuser:12345@10.0.3.253:5431/optuna_db"
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()


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
        if mpi_rank == 0:
            all_time = []
            all_values = []
        for i, objective in enumerate(self.objectives):
            logging.info(f"start {self.args.task} {i}")

            if mpi_rank == 0:
                study = optuna.create_study(
                    study_name=f"{self.args.task}_{i}_N{mpi_size}_n{self.args.n_proc}",
                    storage=URL_DB,
                    load_if_exists=False,
                )
            mpi_comm.Barrier()
            if mpi_rank > 0:
                study = optuna.load_study(
                    study_name=f"{self.args.task}_{i}_N{mpi_size}_n{self.args.n_proc}",
                    storage=URL_DB,
                )

            div, mod = divmod(self.n_iter, self.args.n_proc)
            counts = []
            for n in range(self.args.n_proc):
                if mpi_rank * self.args.n_proc + n < mod:
                    counts.append(div + 1)
                else:
                    counts.append(div)

            with ProcessingPool(self.args.n_proc) as p:
                mpi_comm.Barrier()
                if mpi_rank == 0:
                    start_time = perf_counter()
                p.map(study.optimize, [objective] * self.args.n_proc, counts)
                mpi_comm.Barrier()
                if mpi_rank == 0:
                    all_time.append(perf_counter() - start_time)
                    all_values.append(study.best_value)
                    optuna.delete_study(
                        study_name=f"{self.args.task}_{i}_N{mpi_size}_n{self.args.n_proc}",
                        storage=URL_DB,
                    )

            logging.info(f"solved {self.args.task} {i}")
        if mpi_rank == 0:
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
            index={0: mpi_size * self.args.n_proc},
        )
        df.to_excel(f"{self.args.task}_N{mpi_size}_n{self.args.n_proc}.xlsx", float_format="%.3f")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_proc", type=int, required=True)
    parser.add_argument("-l", "--launches", type=int, required=True)
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
    logging.info(f"my rank {mpi_rank}, size {mpi_size}")
    arguments = parse_arguments()
    ex = OptunaExperiment(arguments)
    ex.calculate_data()
    if mpi_rank == 0:
        ex.save_data()
