from ortools.linear_solver import pywraplp
import numpy as np

# Row i represents how much each person j owes i, summed over all individual purchases covered by i.
n = 4
debts = np.array([
    [0, 10, 10, 10],
    [20, 0, 20, 0],
    [10, 20, 0, 0],
    [0, 0, 100.2, 0]
])

def minimize_payments():
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return
    
    # For each (i, j), create a "positive edge" only when the amount j owes i is non-zero (positive).
    edges = [(i, j) for j in range (n) for i in range(n) if debts[i, j] > 0]

    # Create payment variables representing how much j should pay i only for each positive edge (i, j).
    # This means that if j never owed i any amount, then j will never pay i any amount.
    pay_vars = [solver.NumVar(0, solver.infinity(), f"x_{i}_{j}") for (i, j) in edges]

    # Debt sum owed by/to each person.
    debt_by_sums = np.sum(debts, axis=0)
    debt_to_sums = np.sum(debts, axis=1)

    # Pay sum made by/to each person. Init arrays to zeroes.
    pay_by_sums = [0] * n
    pay_to_sums = [0] * n

    # For edge (i, j), update how much j pays i to each pay sum array.
    for ((i, j), pay_var) in zip(edges, pay_vars):
        pay_by_sums[j] += pay_var
        pay_to_sums[i] += pay_var

    # (Pay sum made BY i minus pay sum made TO i) must equal (debt sum owed BY i minus debt sum owed TO i).
    for i in range(n):
        solver.Add(pay_by_sums[i] - pay_to_sums[i] == debt_by_sums[i] - debt_to_sums[i])

    # Minimize sum of total payments. This also conveniently minimizes the number of non-zero payments.
    solver.Minimize(sum(pay_vars))

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print(f"Objective value = {solver.Objective().Value():0.1f}")

        # Each row i in the solution represents how much i should venmo request each person j.
        soln = np.zeros((n, n))
        for ((i, j), pay_var) in zip(edges, pay_vars):
            soln[i, j] = abs(pay_var.solution_value())
        return soln
    else:
        print("The problem does not have an optimal solution.")
        return None

if __name__ == "__main__":
    print(minimize_payments())
    