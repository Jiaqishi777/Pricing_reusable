"""
Numerical routines for the reusable-resource pricing model.

Dependencies
------------
    numpy
    scipy

The decision variable is the effective arrival rate lambda, not the price.
Each demand dictionary stores a market-size upper bound Lambda and enough
information to evaluate p(lambda) and r(lambda) = lambda * p(lambda).

Main routines
-------------
1. solve_one_class_probability
   Solves the 1-class optimal inventory-based policy by treating the
   steady-state probabilities as decision variables.

2. solve_inventory_rvi
   Solves the multi-class exponential-service model by uniformized relative
   value iteration.

3. solve_inventory_policy_iteration
   Solves the same average-reward dynamic program by policy iteration.
   It is much faster than uniformized RVI for stiff instances such as
   Appendix B, Example 2, where service rates differ by several orders of
   magnitude.

4. solve_optimal_static
   Solves the concave fractional static-pricing problem by Dinkelbach's
   method. Each inner problem is a convex minimization problem because the
   revenue numerator is concave and the reciprocal service level is convex.

5. constructed_tilde and constructed_geometric
   Compute the policies \tilde{lambda} and lambda^g from an optimal
   inventory-based policy.
"""

import math
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.special import logsumexp, xlogy


# ---------------------------------------------------------------------------
# Demand models
# ---------------------------------------------------------------------------

def linear_demand(
    price_intercept: float,
    price_slope: float,
    market_size: Optional[float] = None,
    name: str = "linear",
) -> dict:
    """Return p(lambda) = price_intercept - price_slope * lambda."""
    if price_intercept <= 0 or price_slope <= 0:
        raise ValueError("price_intercept and price_slope must be positive")
    if market_size is None:
        market_size = price_intercept / price_slope
    return {
        "kind": "linear",
        "price_intercept": float(price_intercept),
        "price_slope": float(price_slope),
        "Lambda": float(market_size),
        "name": name,
    }


def exponential_demand(
    market_size: float,
    price_scale: float = 1.0,
    name: str = "exponential",
) -> dict:
    """Return p(lambda) = price_scale * log(market_size / lambda)."""
    if market_size <= 0 or price_scale <= 0:
        raise ValueError("market_size and price_scale must be positive")
    return {
        "kind": "exponential",
        "price_scale": float(price_scale),
        "Lambda": float(market_size),
        "name": name,
    }


def shifted_equal_revenue_demand(
    market_size: float,
    shift: float,
    tail_constant: float = 1.0,
    name: str = "shifted equal revenue",
) -> dict:
    """Return p(lambda) = shift + tail_constant / lambda.

    At lambda = 0, the code uses the upper-semicontinuous closure
    r(0) = lim_{lambda -> 0} lambda p(lambda) = tail_constant, as in the
    manuscript's convention for unbounded valuation support.
    """
    if market_size <= 0 or shift < 0 or tail_constant < 0:
        raise ValueError("invalid shifted equal-revenue parameters")
    return {
        "kind": "shifted_equal_revenue",
        "shift": float(shift),
        "tail_constant": float(tail_constant),
        "Lambda": float(market_size),
        "name": name,
    }


def custom_demand(
    market_size: float,
    price_function: Callable[[float], float],
    revenue_function: Optional[Callable[[float], float]] = None,
    name: str = "custom",
) -> dict:
    """Create a custom demand model.

    The supplied revenue function should be concave on [0, market_size].
    If it is omitted, the code uses lambda * price_function(lambda).
    """
    if market_size <= 0:
        raise ValueError("market_size must be positive")
    if revenue_function is None:
        revenue_function = lambda lam: lam * price_function(lam)
    return {
        "kind": "custom",
        "price_function": price_function,
        "revenue_function": revenue_function,
        "Lambda": float(market_size),
        "name": name,
    }


def price(demand: dict, lam: float) -> float:
    """Price corresponding to an effective arrival rate."""
    lam = float(lam)
    kind = demand["kind"]
    if kind == "linear":
        return demand["price_intercept"] - demand["price_slope"] * lam
    if kind == "exponential":
        if lam <= 0:
            return math.inf
        return demand["price_scale"] * math.log(demand["Lambda"] / lam)
    if kind == "shifted_equal_revenue":
        if lam <= 0:
            return math.inf
        return demand["shift"] + demand["tail_constant"] / lam
    if kind == "custom":
        return float(demand["price_function"](lam))
    raise ValueError(f"unknown demand kind: {kind}")


def revenue_rate(demand: dict, lam: float) -> float:
    """Revenue rate r(lambda) = lambda p(lambda)."""
    lam = float(np.clip(lam, 0.0, demand["Lambda"]))
    kind = demand["kind"]
    if kind == "linear":
        return (
            demand["price_intercept"] * lam
            - demand["price_slope"] * lam * lam
        )
    if kind == "exponential":
        if lam <= 0:
            return 0.0
        return demand["price_scale"] * lam * math.log(demand["Lambda"] / lam)
    if kind == "shifted_equal_revenue":
        return demand["shift"] * lam + demand["tail_constant"]
    if kind == "custom":
        return float(demand["revenue_function"](lam))
    raise ValueError(f"unknown demand kind: {kind}")


def monopoly_rate(demand: dict) -> float:
    """Maximizer of r(lambda) on [0, Lambda]."""
    kind = demand["kind"]
    if kind == "linear":
        value = demand["price_intercept"] / (2.0 * demand["price_slope"])
        return float(np.clip(value, 0.0, demand["Lambda"]))
    if kind == "exponential":
        return demand["Lambda"] / math.e
    if kind == "shifted_equal_revenue":
        if demand["shift"] > 0:
            return demand["Lambda"]
        return 0.0

    result = minimize_scalar(
        lambda lam: -revenue_rate(demand, lam),
        bounds=(0.0, demand["Lambda"]),
        method="bounded",
        options={"xatol": 1e-12},
    )
    candidates = [0.0, float(result.x), demand["Lambda"]]
    return max(candidates, key=lambda lam: revenue_rate(demand, lam))


def best_rate_with_value_adjustment(
    demand: dict,
    value_adjustment: float,
    upper_bound: Optional[float] = None,
) -> float:
    """Maximize r(lambda) + value_adjustment * lambda.

    In the average optimality equation, value_adjustment is
    h(next state) - h(current state).
    """
    if upper_bound is None:
        upper_bound = demand["Lambda"]
    upper_bound = min(float(upper_bound), demand["Lambda"])
    kind = demand["kind"]

    if kind == "linear":
        value = (
            demand["price_intercept"] + value_adjustment
        ) / (2.0 * demand["price_slope"])
        return float(np.clip(value, 0.0, upper_bound))

    if kind == "exponential":
        exponent = value_adjustment / demand["price_scale"] - 1.0
        exponent = float(np.clip(exponent, -745.0, 700.0))
        value = demand["Lambda"] * math.exp(exponent)
        return float(np.clip(value, 0.0, upper_bound))

    if kind == "shifted_equal_revenue":
        coefficient = demand["shift"] + value_adjustment
        return upper_bound if coefficient > 0 else 0.0

    objective = lambda lam: -(
        revenue_rate(demand, lam) + value_adjustment * lam
    )
    result = minimize_scalar(
        objective,
        bounds=(0.0, upper_bound),
        method="bounded",
        options={"xatol": 1e-12},
    )
    candidates = [0.0, float(result.x), upper_bound]
    return min(candidates, key=objective)


# ---------------------------------------------------------------------------
# Erlang-loss calculations and static policy evaluation
# ---------------------------------------------------------------------------

def erlang_b(C: int, offered_load: float) -> float:
    """Stable recursion for the Erlang-B blocking probability."""
    if C < 1:
        raise ValueError("C must be at least 1")
    if offered_load < 0:
        raise ValueError("offered_load must be nonnegative")
    blocking = 1.0
    for c in range(1, C + 1):
        if offered_load == 0:
            blocking = 0.0
        else:
            blocking = offered_load * blocking / (c + offered_load * blocking)
    return blocking


def static_service_level(C: int, offered_load: float) -> float:
    return 1.0 - erlang_b(C, offered_load)


def static_revenue(
    C: int,
    demands: list[dict],
    service_rates: np.ndarray,
    arrival_rates: np.ndarray,
) -> float:
    """Revenue of a class-dependent static policy."""
    arrival_rates = np.asarray(arrival_rates, dtype=float)
    service_rates = np.asarray(service_rates, dtype=float)
    offered_load = float(np.sum(arrival_rates / service_rates))
    instantaneous_revenue = sum(
        revenue_rate(demands[j], arrival_rates[j])
        for j in range(len(demands))
    )
    return instantaneous_revenue * static_service_level(C, offered_load)


def reciprocal_service_level(C: int, offered_load: float) -> float:
    service_level = static_service_level(C, offered_load)
    return 1.0 / max(service_level, 1e-300)


def solve_optimal_static(
    C: int,
    demands: list[dict],
    service_rates: np.ndarray,
    tolerance: float = 1e-11,
    max_iterations: int = 200,
) -> dict:
    """Solve the optimal static concave fractional program.

    The objective is
        sum_j r_j(lambda_j) / Q_C(sum_j lambda_j / mu_j),
    where Q_C is the reciprocal Erlang service level.

    Dinkelbach's method repeatedly solves
        max sum_j r_j(lambda_j) - q Q_C(sum_j lambda_j / mu_j).
    The inner objective is concave, so minimizing its negative over box
    constraints is a convex optimization problem.
    """
    service_rates = np.asarray(service_rates, dtype=float)
    if len(demands) != len(service_rates):
        raise ValueError("demands and service_rates must have the same length")
    if np.any(service_rates <= 0):
        raise ValueError("service rates must be positive")

    upper_bounds = np.array([monopoly_rate(d) for d in demands], dtype=float)
    bounds = [(0.0, upper_bounds[j]) for j in range(len(demands))]
    arrival_rates = np.maximum(0.5 * upper_bounds, 1e-8)
    arrival_rates = np.minimum(arrival_rates, upper_bounds)

    numerator = sum(
        revenue_rate(demands[j], arrival_rates[j])
        for j in range(len(demands))
    )
    offered_load = float(np.sum(arrival_rates / service_rates))
    denominator = reciprocal_service_level(C, offered_load)
    ratio = numerator / denominator

    inner_result = None
    residual = math.inf
    for iteration in range(max_iterations):
        def negative_inner_objective(values: np.ndarray) -> float:
            value_numerator = sum(
                revenue_rate(demands[j], values[j])
                for j in range(len(demands))
            )
            value_load = float(np.sum(values / service_rates))
            value_denominator = reciprocal_service_level(C, value_load)
            return -(value_numerator - ratio * value_denominator)

        inner_result = minimize(
            negative_inner_objective,
            arrival_rates,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-14,
                "gtol": 1e-10,
                "maxiter": 10000,
                "maxls": 50,
            },
        )
        arrival_rates = np.asarray(inner_result.x, dtype=float)
        numerator = sum(
            revenue_rate(demands[j], arrival_rates[j])
            for j in range(len(demands))
        )
        offered_load = float(np.sum(arrival_rates / service_rates))
        denominator = reciprocal_service_level(C, offered_load)
        residual = numerator - ratio * denominator
        ratio = numerator / denominator
        if abs(residual) <= tolerance:
            break

    return {
        "arrival_rates": arrival_rates,
        "prices": np.array([price(demands[j], arrival_rates[j]) for j in range(len(demands))]),
        "revenue": static_revenue(C, demands, service_rates, arrival_rates),
        "service_level": static_service_level(C, offered_load),
        "offered_load": offered_load,
        "dinkelbach_iterations": iteration + 1,
        "dinkelbach_residual": residual,
        "inner_success": bool(inner_result.success) if inner_result is not None else False,
        "inner_message": str(inner_result.message) if inner_result is not None else "",
    }


# ---------------------------------------------------------------------------
# State enumeration and CTMC calculations
# ---------------------------------------------------------------------------

def enumerate_states(C: int, M: int) -> list[tuple[int, ...]]:
    """All x in Z_+^M with sum_j x_j <= C."""
    states: list[tuple[int, ...]] = []

    def add_states(prefix: list[int], remaining_capacity: int) -> None:
        if len(prefix) == M - 1:
            for final_count in range(remaining_capacity + 1):
                states.append(tuple(prefix + [final_count]))
            return
        for count in range(remaining_capacity + 1):
            add_states(prefix + [count], remaining_capacity - count)

    add_states([], C)
    return states


def build_generator(
    C: int,
    states: list[tuple[int, ...]],
    policy: np.ndarray,
    service_rates: np.ndarray,
) -> np.ndarray:
    """CTMC generator under a fixed inventory-based policy."""
    service_rates = np.asarray(service_rates, dtype=float)
    M = len(service_rates)
    state_index = {state: index for index, state in enumerate(states)}
    generator = np.zeros((len(states), len(states)))

    for state_number, state in enumerate(states):
        if sum(state) < C:
            for j in range(M):
                next_state = list(state)
                next_state[j] += 1
                generator[state_number, state_index[tuple(next_state)]] += policy[state_number, j]

        for j in range(M):
            if state[j] > 0:
                previous_state = list(state)
                previous_state[j] -= 1
                departure_rate = state[j] * service_rates[j]
                generator[state_number, state_index[tuple(previous_state)]] += departure_rate

        generator[state_number, state_number] = -np.sum(generator[state_number])

    return generator


def stationary_distribution(generator: np.ndarray) -> np.ndarray:
    """Solve pi Q = 0 and sum pi = 1."""
    number_of_states = generator.shape[0]
    matrix = generator.T.copy()
    right_hand_side = np.zeros(number_of_states)
    matrix[-1, :] = 1.0
    right_hand_side[-1] = 1.0

    try:
        probabilities = np.linalg.solve(matrix, right_hand_side)
    except np.linalg.LinAlgError:
        probabilities = np.linalg.lstsq(matrix, right_hand_side, rcond=None)[0]

    probabilities[np.abs(probabilities) < 1e-14] = 0.0
    if np.min(probabilities) < -1e-9:
        raise RuntimeError("stationary distribution has a materially negative entry")
    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= np.sum(probabilities)
    return probabilities


def policy_revenue(
    C: int,
    states: list[tuple[int, ...]],
    probabilities: np.ndarray,
    policy: np.ndarray,
    demands: list[dict],
) -> float:
    total = 0.0
    for state_number, state in enumerate(states):
        if sum(state) < C:
            total += probabilities[state_number] * sum(
                revenue_rate(demands[j], policy[state_number, j])
                for j in range(len(demands))
            )
    return total


# ---------------------------------------------------------------------------
# 1-class optimization using steady-state probabilities
# ---------------------------------------------------------------------------

def _one_class_perspective_revenue(
    demand: dict,
    stationary_flow: float,
    probability: float,
) -> float:
    """probability * r(stationary_flow / probability)."""
    probability = max(float(probability), 1e-300)
    stationary_flow = max(float(stationary_flow), 0.0)
    kind = demand["kind"]

    if kind == "linear":
        return (
            demand["price_intercept"] * stationary_flow
            - demand["price_slope"] * stationary_flow * stationary_flow / probability
        )

    if kind == "exponential":
        if stationary_flow <= 0:
            return 0.0
        return demand["price_scale"] * (
            stationary_flow * math.log(demand["Lambda"] * probability)
            - xlogy(stationary_flow, stationary_flow)
        )

    if kind == "shifted_equal_revenue":
        return (
            demand["shift"] * stationary_flow
            + demand["tail_constant"] * probability
        )

    effective_arrival_rate = stationary_flow / probability
    return probability * revenue_rate(demand, effective_arrival_rate)


def _birth_death_probabilities(
    policy: np.ndarray,
    service_rate: float,
) -> np.ndarray:
    """Stationary probabilities implied by lambda_0,...,lambda_{C-1}."""
    C = len(policy)
    log_weights = np.full(C + 1, -math.inf)
    log_weights[0] = 0.0
    for i in range(C):
        if policy[i] <= 0 or not np.isfinite(log_weights[i]):
            break
        log_weights[i + 1] = (
            log_weights[i]
            + math.log(policy[i])
            - math.log((i + 1) * service_rate)
        )
    normalizer = logsumexp(log_weights)
    return np.exp(log_weights - normalizer)


def solve_one_class_probability(
    C: int,
    demand: dict,
    service_rate: float,
    tolerance: float = 1e-12,
    max_iterations: int = 10000,
) -> dict:
    """Optimal 1-class inventory-based policy via steady-state probabilities.

    Let P_i be the steady-state probability of i occupied units. The balance
    equations imply
        lambda_i P_i = (i + 1) mu P_{i+1}.
    Hence the objective is
        sum_{i=0}^{C-1} P_i r((i+1) mu P_{i+1} / P_i).
    Each term is the perspective of the concave revenue rate r, so the
    objective is concave in P. The constraints are linear:
        sum_i P_i = 1,
        (i+1) mu P_{i+1} <= Lambda P_i,
        P_i >= 0.

    The shifted equal-revenue model gives a linear program and is solved by
    scipy.optimize.linprog. Other built-in models are solved by SLSQP.
    """
    if C < 1:
        raise ValueError("C must be at least 1")
    if service_rate <= 0:
        raise ValueError("service_rate must be positive")

    if demand["kind"] == "shifted_equal_revenue":
        linear_objective = np.zeros(C + 1)
        for i in range(C):
            linear_objective[i] -= demand["tail_constant"]
            linear_objective[i + 1] -= demand["shift"] * (i + 1) * service_rate

        inequality_matrix = []
        inequality_rhs = []
        for i in range(C):
            row = np.zeros(C + 1)
            row[i + 1] = (i + 1) * service_rate
            row[i] = -demand["Lambda"]
            inequality_matrix.append(row)
            inequality_rhs.append(0.0)

        result = linprog(
            linear_objective,
            A_ub=np.asarray(inequality_matrix),
            b_ub=np.asarray(inequality_rhs),
            A_eq=np.ones((1, C + 1)),
            b_eq=np.array([1.0]),
            bounds=[(0.0, None)] * (C + 1),
            method="highs",
        )
        optimized_probabilities = np.asarray(result.x, dtype=float)
        solver_success = bool(result.success)
        solver_message = str(result.message)
        solver_objective = -float(result.fun)
    else:
        initial_rate = min(
            max(0.5 * C * service_rate, 1e-8),
            max(monopoly_rate(demand), 1e-8),
        )
        initial_policy = np.full(C, initial_rate)
        initial_probabilities = _birth_death_probabilities(
            initial_policy,
            service_rate,
        )

        def objective(probabilities: np.ndarray) -> float:
            value = 0.0
            for i in range(C):
                stationary_flow = (i + 1) * service_rate * probabilities[i + 1]
                value += _one_class_perspective_revenue(
                    demand,
                    stationary_flow,
                    probabilities[i],
                )
            return -value

        constraints = [
            {
                "type": "eq",
                "fun": lambda probabilities: np.sum(probabilities) - 1.0,
            }
        ]
        for i in range(C):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda probabilities, i=i: (
                        demand["Lambda"] * probabilities[i]
                        - (i + 1) * service_rate * probabilities[i + 1]
                    ),
                }
            )

        result = minimize(
            objective,
            initial_probabilities,
            method="SLSQP",
            bounds=[(1e-16, 1.0)] * (C + 1),
            constraints=constraints,
            options={
                "ftol": tolerance,
                "maxiter": max_iterations,
                "disp": False,
            },
        )
        optimized_probabilities = np.asarray(result.x, dtype=float)
        optimized_probabilities = np.maximum(optimized_probabilities, 0.0)
        optimized_probabilities /= np.sum(optimized_probabilities)
        solver_success = bool(result.success)
        solver_message = str(result.message)
        solver_objective = -float(result.fun)

    policy_vector = np.zeros(C)
    for i in range(C):
        if optimized_probabilities[i] > 1e-300:
            policy_vector[i] = (
                (i + 1) * service_rate * optimized_probabilities[i + 1]
                / optimized_probabilities[i]
            )
    policy_vector = np.clip(policy_vector, 0.0, demand["Lambda"])

    # Recompute the stationary probabilities from the recovered policy so the
    # reported policy, probabilities, and revenue are exactly consistent.
    probabilities = _birth_death_probabilities(policy_vector, service_rate)
    states = [(i,) for i in range(C + 1)]
    policy = np.zeros((C + 1, 1))
    policy[:C, 0] = policy_vector
    revenue = sum(
        probabilities[i] * revenue_rate(demand, policy_vector[i])
        for i in range(C)
    )

    max_balance_error = max(
        abs(
            policy_vector[i] * probabilities[i]
            - (i + 1) * service_rate * probabilities[i + 1]
        )
        for i in range(C)
    )

    return {
        "method": "steady-state probability optimization",
        "states": states,
        "probabilities": probabilities,
        "policy": policy,
        "revenue": revenue,
        "prices": np.array([price(demand, policy_vector[i]) for i in range(C)]),
        "solver_success": solver_success,
        "solver_message": solver_message,
        "solver_objective_before_recovery": solver_objective,
        "max_balance_error": max_balance_error,
    }


# ---------------------------------------------------------------------------
# Multi-class average dynamic programming
# ---------------------------------------------------------------------------

def _evaluate_policy_average_reward(
    C: int,
    states: list[tuple[int, ...]],
    policy: np.ndarray,
    demands: list[dict],
    service_rates: np.ndarray,
    reference_state_number: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Solve the Poisson equation for a fixed policy."""
    generator = build_generator(C, states, policy, service_rates)
    rewards = np.zeros(len(states))
    for state_number, state in enumerate(states):
        if sum(state) < C:
            rewards[state_number] = sum(
                revenue_rate(demands[j], policy[state_number, j])
                for j in range(len(demands))
            )

    # Unknowns are h_0,...,h_{N-1}, and the average reward g.
    # The equations are -Q h + g 1 = r, together with h(reference)=0.
    number_of_states = len(states)
    matrix = np.zeros((number_of_states + 1, number_of_states + 1))
    right_hand_side = np.zeros(number_of_states + 1)
    matrix[:number_of_states, :number_of_states] = -generator
    matrix[:number_of_states, number_of_states] = 1.0
    right_hand_side[:number_of_states] = rewards
    matrix[number_of_states, reference_state_number] = 1.0

    solution = np.linalg.solve(matrix, right_hand_side)
    relative_values = solution[:number_of_states]
    average_reward = float(solution[number_of_states])
    return relative_values, average_reward, generator


def solve_inventory_policy_iteration(
    C: int,
    demands: list[dict],
    service_rates: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> dict:
    """Average-reward policy iteration for exponential service times.

    This solves the same average dynamic program as relative value iteration,
    but it is usually much faster when transition rates are highly unequal.
    """
    service_rates = np.asarray(service_rates, dtype=float)
    M = len(demands)
    if M != len(service_rates):
        raise ValueError("demands and service_rates must have the same length")
    if np.any(service_rates <= 0):
        raise ValueError("service rates must be positive")

    states = enumerate_states(C, M)
    state_index = {state: index for index, state in enumerate(states)}
    empty_state = tuple([0] * M)
    reference_state_number = state_index[empty_state]

    policy = np.zeros((len(states), M))
    for state_number, state in enumerate(states):
        if sum(state) < C:
            policy[state_number, :] = 0.5 * np.array(
                [monopoly_rate(demand) for demand in demands]
            )

    policy_change = math.inf
    for iteration in range(max_iterations):
        relative_values, average_reward, generator = _evaluate_policy_average_reward(
            C,
            states,
            policy,
            demands,
            service_rates,
            reference_state_number,
        )

        improved_policy = np.zeros_like(policy)
        for state_number, state in enumerate(states):
            if sum(state) >= C:
                continue
            for j in range(M):
                next_state = list(state)
                next_state[j] += 1
                next_state_number = state_index[tuple(next_state)]
                value_adjustment = (
                    relative_values[next_state_number]
                    - relative_values[state_number]
                )
                improved_policy[state_number, j] = best_rate_with_value_adjustment(
                    demands[j],
                    value_adjustment,
                    demands[j]["Lambda"],
                )

        policy_change = float(np.max(np.abs(improved_policy - policy)))
        policy = improved_policy
        if policy_change <= tolerance:
            break

    relative_values, average_reward, generator = _evaluate_policy_average_reward(
        C,
        states,
        policy,
        demands,
        service_rates,
        reference_state_number,
    )
    probabilities = stationary_distribution(generator)
    revenue = policy_revenue(C, states, probabilities, policy, demands)

    return {
        "method": "average-reward policy iteration",
        "states": states,
        "probabilities": probabilities,
        "policy": policy,
        "revenue": revenue,
        "average_reward_from_poisson_equation": average_reward,
        "relative_values": relative_values,
        "iterations": iteration + 1,
        "policy_change": policy_change,
        "converged": policy_change <= tolerance,
    }


def solve_inventory_rvi(
    C: int,
    demands: list[dict],
    service_rates: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 200000,
    action_upper_bounds: Optional[np.ndarray] = None,
) -> dict:
    """Uniformized relative value iteration for exponential service times.

    The default uniformization rate is
        sum_j Lambda_j + C max_j mu_j.
    This is valid but may be conservative. For a stiff instance, RVI can be
    slow. The policy-iteration routine above is preferable in that case.

    action_upper_bounds may be supplied to reduce the uniformization rate,
    but each bound must remain at least as large as the optimal action.
    """
    service_rates = np.asarray(service_rates, dtype=float)
    M = len(demands)
    if M != len(service_rates):
        raise ValueError("demands and service_rates must have the same length")
    if np.any(service_rates <= 0):
        raise ValueError("service rates must be positive")

    if action_upper_bounds is None:
        action_upper_bounds = np.array(
            [demand["Lambda"] for demand in demands],
            dtype=float,
        )
    else:
        action_upper_bounds = np.asarray(action_upper_bounds, dtype=float)
        if len(action_upper_bounds) != M:
            raise ValueError("action_upper_bounds must have length M")
        for j in range(M):
            if action_upper_bounds[j] <= 0 or action_upper_bounds[j] > demands[j]["Lambda"]:
                raise ValueError("invalid action upper bound")

    states = enumerate_states(C, M)
    state_index = {state: index for index, state in enumerate(states)}
    empty_state = tuple([0] * M)
    reference_state_number = state_index[empty_state]
    uniformization_rate = float(
        np.sum(action_upper_bounds) + C * np.max(service_rates)
    )

    relative_values = np.zeros(len(states))
    span_residual = math.inf
    converged = False

    for iteration in range(max_iterations):
        updated_values = np.empty_like(relative_values)
        for state_number, state in enumerate(states):
            drift_value = 0.0

            if sum(state) < C:
                for j in range(M):
                    next_state = list(state)
                    next_state[j] += 1
                    next_state_number = state_index[tuple(next_state)]
                    value_adjustment = (
                        relative_values[next_state_number]
                        - relative_values[state_number]
                    )
                    lam = best_rate_with_value_adjustment(
                        demands[j],
                        value_adjustment,
                        action_upper_bounds[j],
                    )
                    drift_value += (
                        revenue_rate(demands[j], lam)
                        + lam * value_adjustment
                    )

            for j in range(M):
                if state[j] > 0:
                    previous_state = list(state)
                    previous_state[j] -= 1
                    previous_state_number = state_index[tuple(previous_state)]
                    departure_rate = state[j] * service_rates[j]
                    drift_value += departure_rate * (
                        relative_values[previous_state_number]
                        - relative_values[state_number]
                    )

            updated_values[state_number] = (
                relative_values[state_number]
                + drift_value / uniformization_rate
            )

        updated_values -= updated_values[reference_state_number]
        increment = updated_values - relative_values
        span_residual = float(np.max(increment) - np.min(increment))
        relative_values = updated_values
        if span_residual <= tolerance:
            converged = True
            break

    policy = np.zeros((len(states), M))
    for state_number, state in enumerate(states):
        if sum(state) >= C:
            continue
        for j in range(M):
            next_state = list(state)
            next_state[j] += 1
            next_state_number = state_index[tuple(next_state)]
            value_adjustment = (
                relative_values[next_state_number]
                - relative_values[state_number]
            )
            policy[state_number, j] = best_rate_with_value_adjustment(
                demands[j],
                value_adjustment,
                action_upper_bounds[j],
            )

    generator = build_generator(C, states, policy, service_rates)
    probabilities = stationary_distribution(generator)
    revenue = policy_revenue(C, states, probabilities, policy, demands)

    return {
        "method": "uniformized relative value iteration",
        "states": states,
        "probabilities": probabilities,
        "policy": policy,
        "revenue": revenue,
        "relative_values": relative_values,
        "iterations": iteration + 1,
        "span_residual": span_residual,
        "uniformization_rate": uniformization_rate,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Constructed policies and performance comparison
# ---------------------------------------------------------------------------

def constructed_tilde(C: int, inventory_result: dict) -> np.ndarray:
    """Compute \tilde{lambda} from an inventory-based policy."""
    states = inventory_result["states"]
    probabilities = np.asarray(inventory_result["probabilities"], dtype=float)
    policy = np.asarray(inventory_result["policy"], dtype=float)
    nonfull_probability = sum(
        probabilities[state_number]
        for state_number, state in enumerate(states)
        if sum(state) < C
    )
    if nonfull_probability <= 0:
        return np.zeros(policy.shape[1])
    return np.sum(probabilities[:, None] * policy, axis=0) / nonfull_probability


def constructed_geometric(C: int, inventory_result: dict) -> float:
    """Compute lambda^g for a 1-class inventory-based policy."""
    policy = np.asarray(inventory_result["policy"], dtype=float)
    if policy.shape[1] != 1:
        raise ValueError("lambda^g is defined here only for the 1-class system")
    states = inventory_result["states"]
    probabilities = np.asarray(inventory_result["probabilities"], dtype=float)
    rates = policy[:, 0]

    flow_weights = np.array(
        [
            probabilities[state_number] * rates[state_number]
            if sum(state) < C
            else 0.0
            for state_number, state in enumerate(states)
        ]
    )
    total_flow = float(np.sum(flow_weights))
    if total_flow <= 0:
        return 0.0

    positive = rates > 0
    return float(
        math.exp(np.sum(flow_weights[positive] * np.log(rates[positive])) / total_flow)
    )


def compare_with_inventory_benchmark(
    C: int,
    demands: list[dict],
    service_rates: np.ndarray,
    inventory_result: dict,
    compute_optimal_static: bool = True,
) -> dict:
    """Evaluate \tilde{lambda}, lambda^g, and the optimal static policy."""
    benchmark_revenue = float(inventory_result["revenue"])
    tilde_rates = constructed_tilde(C, inventory_result)
    tilde_revenue = static_revenue(C, demands, service_rates, tilde_rates)

    comparison = {
        "inventory_revenue": benchmark_revenue,
        "tilde_rates": tilde_rates,
        "tilde_prices": np.array([price(demands[j], tilde_rates[j]) for j in range(len(demands))]),
        "tilde_revenue": tilde_revenue,
        "tilde_ratio": tilde_revenue / benchmark_revenue,
    }

    if len(demands) == 1:
        geometric_rate = constructed_geometric(C, inventory_result)
        geometric_revenue = static_revenue(
            C,
            demands,
            service_rates,
            np.array([geometric_rate]),
        )
        comparison.update(
            {
                "geometric_rate": geometric_rate,
                "geometric_price": price(demands[0], geometric_rate),
                "geometric_revenue": geometric_revenue,
                "geometric_ratio": geometric_revenue / benchmark_revenue,
            }
        )

    if compute_optimal_static:
        optimal_static = solve_optimal_static(C, demands, service_rates)
        comparison["optimal_static"] = optimal_static
        comparison["optimal_static_ratio"] = (
            optimal_static["revenue"] / benchmark_revenue
        )

    return comparison


def two_class_policy_matrix(
    C: int,
    inventory_result: dict,
    class_number: int,
) -> np.ndarray:
    """Triangular matrix indexed by (x_1, x_2) for a 2-class policy."""
    if inventory_result["policy"].shape[1] != 2:
        raise ValueError("this helper requires exactly two classes")
    if class_number not in (0, 1):
        raise ValueError("class_number must be 0 or 1")

    matrix = np.full((C + 1, C + 1), np.nan)
    for state_number, state in enumerate(inventory_result["states"]):
        if sum(state) < C:
            matrix[state[0], state[1]] = inventory_result["policy"][state_number, class_number]
    return matrix
