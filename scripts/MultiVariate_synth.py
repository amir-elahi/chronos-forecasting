import numpy as np
import random
import importlib
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from pathlib import Path
import argparse
import h5py

# Dynamically import 'kernel-synth' from the 'scripts' directory
kernel_synth = importlib.import_module("kernel-synth")


def generate_multiVariate_time_series(
        max_num_operations: int,
        max_num_kernels: int = 5,
        operations: list[str] = ['+', '*'],
        lr_lambda: float = -2.0,
        hr_lambda: float = 2.0,

):

    context = list(
        kernel_synth.generate_time_series(max_kernels=max_num_kernels) for _ in range(3)
    )

    a = context[0]['target']
    b = context[1]['target']
    c = context[2]['target']

    number_of_operations = random.randint(0, max_num_operations)
    operation_list = [random.choice(operations) for _ in range(number_of_operations)]
    lambda_list = [random.uniform(lr_lambda, hr_lambda) for _ in range(number_of_operations)]

    # Extra random constant to add at the end
    extra_random_value = random.uniform(lr_lambda, hr_lambda)

    # Testing setup to see the resulting expression
    if len(operation_list) == 0:
        target = c + extra_random_value
    else:
        expression = np.zeros_like(a, dtype=float)
        current_term = None  # Start with no term defined

        for op, lambda_value in zip(operation_list, lambda_list):
            operand = random.choice(['a', 'b'])
            chosen_operand = a if operand == 'a' else b

            if op == '*':
                if current_term is None:
                    current_term = lambda_value * chosen_operand
                else:
                    current_term *= lambda_value * chosen_operand
            elif op == '+':
                if current_term is not None:
                    expression += current_term  # Add previous term to expression

                # Start a new term
                current_term = lambda_value * chosen_operand

        # Add the last term if any
        if current_term is not None:
            expression += current_term

        # Add the extra random value to both the formula and the target expression
        target = expression + extra_random_value

        target = target.reshape(1, -1)
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    return [target, a, b]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-series", type=int, default=1_000_000)
    parser.add_argument("-O", "--max-num-operations", type=int, default=10)
    parser.add_argument("-L", "--lr_lambda", type=float, default=-2.0)
    parser.add_argument('-H', "--hr_lambda", type=float, default=2.0)
    parser.add_argument('-NK', "--num_kernels", type=int, default=5)
    parser.add_argument('--name', type=str, default='MultivariateData')

    args = parser.parse_args()
    path = Path(__file__).parent / \
    f'{args.name}_{args.num_series}_{args.max_num_operations}_{args.num_kernels}.h5'

    generated_dataset = Parallel(n_jobs=-1)(
        delayed(generate_multiVariate_time_series)(
            max_num_operations=args.max_num_operations,
            max_num_kernels=args.num_kernels,
            lr_lambda=args.lr_lambda,
            hr_lambda=args.hr_lambda)
        for _ in tqdm(range(args.num_series))
    )

    with h5py.File(path, 'w') as f:
        for i, inner_list in enumerate(generated_dataset):
            # Store the arrays directly
            f.create_dataset(f"dataset_{i}", data=inner_list)
