"""Generates offspring teams by mutation and crossover (rates set by the user).

Reproduction is the only class that creates or alters teams, so it owns every check
on them: each team it returns is one of the topology's allowed teams, and every
blueprint on it names a configured model with parameters from that model's space.
"""

from typeguard import typechecked
from typing import List, Tuple, Dict, Any
import numpy as np

from ..Agents.blueprint import BluePrint
from ..Team_Builder.team import Team
from ..Team_Builder.topology import SUPERVISOR, TopologyRules
from ..Models.model_config import ModelConfig


@typechecked
class Reproduction:
    def __init__(self,
                 topology: TopologyRules,
                 model_config: ModelConfig,
                 crossover_prob: float,
                 prob_mut_blueprint: float,
                 prob_mut_parameters: float,
                 mut_parameters_cnt: int,
                 prob_mut_topology: float) -> None:
        """
        Holds everything fixed for the whole run that shapes a new team: the
        allowed teams, the models, and the variation knobs (operator probabilities
        and the mutation count).

        Parameters:
            topology (TopologyRules): The run's checked topology rules; every team
                this class returns is one of ``topology.allowed_teams``.
            model_config (ModelConfig): The run's models. Its model pool is the set
                of models a blueprint may use, and each model's param_space is the
                set of values its parameters may take.
            crossover_prob (float): Probability that an offspring is produced by
                crossover; otherwise it is produced by mutation. A crossover child
                is then mutated with probability 1 - crossover_prob.
            prob_mut_blueprint (float): Per-agent probability of the "whole new
                blueprint" mutation outcome (new model + fresh parameters).
            prob_mut_parameters (float): Per-agent probability of the "mutate
                mut_parameters_cnt parameters" outcome. With prob_mut_blueprint it
                must sum to <= 1; the remainder keeps the agent unchanged.
            mut_parameters_cnt (int): Parameters swapped by the "mutate parameters"
                outcome (clamped per model to what it exposes).
            prob_mut_topology (float): Probability of a topology-level mutation,
                which adds or removes exactly one teammate.
        """
        assert 0 < mut_parameters_cnt, f"Number of parameters to mutate must be greater than 0: {mut_parameters_cnt}."
        assert 0.0 <= prob_mut_topology <= 1.0, f"prob_mut_topology must be between 0 and 1: {prob_mut_topology}."
        assert 0.0 <= crossover_prob <= 1.0, f"Crossover probability must be between 0 and 1: {crossover_prob}."
        assert 0.0 <= prob_mut_blueprint <= 1.0, f"prob_mut_blueprint must be between 0 and 1: {prob_mut_blueprint}."
        assert 0.0 <= prob_mut_parameters <= 1.0, f"prob_mut_parameters must be between 0 and 1: {prob_mut_parameters}."
        # The two agent-level mutation outcomes plus "keep unchanged" partition the
        # probability mass, so their sum cannot exceed 1.
        assert prob_mut_blueprint + prob_mut_parameters <= 1.0, (
            f"prob_mut_blueprint + prob_mut_parameters must be <= 1: "
            f"{prob_mut_blueprint} + {prob_mut_parameters}.")
        self.crossover_prob = float(crossover_prob)
        self.prob_mut_blueprint = float(prob_mut_blueprint)
        self.prob_mut_parameters = float(prob_mut_parameters)
        self.mut_parameters_cnt = int(mut_parameters_cnt)
        self.prob_mut_topology = float(prob_mut_topology)

        self.topology = topology
        self.model_config = model_config
        self.model_pool: List[str] = model_config.model_pool()
        assert len(self.model_pool) > 0, (
            f"{model_config.source} configures no models, so there is nothing to "
            f"build a blueprint from.")

        # Offspring whose composition was not allowed and had to be moved to the
        # nearest allowed team. The EA reads and resets this each generation; a
        # high count means the restrictions and the operators are fighting.
        self.repair_count = 0
        return

    # ---------------------------------------------------------------- checks

    def validate_team(self, team: Team) -> None:
        """Fail unless ``team`` is an allowed composition of valid blueprints."""
        roster = team.get_roster()
        assert SUPERVISOR in roster, f"Team has no '{SUPERVISOR}': {sorted(roster)}."
        assert self.topology.is_allowed(roster), (
            f"Team {sorted(roster)} is not an allowed composition.")
        for role, blueprint in roster.items():
            self.validate_blueprint(role, blueprint)

    def validate_blueprint(self, role: str, blueprint: BluePrint) -> None:
        """Fail unless ``blueprint`` names a configured model and sets exactly that
        model's parameters, each to one of its allowed values. A model with
        ``param_space: {}`` must therefore carry ``{}``."""
        model = blueprint.get_model()
        assert model in self.model_pool, (
            f"'{role}' uses model '{model}', which is not in the model pool "
            f"{self.model_pool}.")
        space = self.model_config.param_space(model)
        missing = sorted(set(space) - set(blueprint.get_params()))
        assert not missing, (
            f"'{role}' does not set {missing}, which the param_space of "
            f"'{model}' requires.")
        for param, value in blueprint.get_params().items():
            assert param in space, (
                f"'{role}' sets '{param}', which is not in the param_space of "
                f"'{model}': {sorted(space)}.")
            assert value in space[param], (
                f"'{role}' sets '{param}' = {value!r}, not one of {space[param]} "
                f"for '{model}'.")

    # ------------------------------------------------------ random generation

    @staticmethod
    def _draw(rng: np.random.Generator, values: List[Any]) -> Any:
        # Index rather than rng.choice(values), which would hand back numpy scalars
        # (np.str_, np.int64) instead of the JSON values the config holds.
        return values[int(rng.integers(len(values)))]

    def generate_random_blueprint(self, rng: np.random.Generator) -> BluePrint:
        """A random model from the model pool with random values for its parameters."""
        model = self._draw(rng, self.model_pool)
        return BluePrint(model=model,
                         parameters=self.generate_random_model_parameters(rng, model))

    def generate_random_model_parameters(self, rng: np.random.Generator, model: str) -> Dict[str, Any]:
        """A random value for every parameter in ``model``'s param_space."""
        assert model in self.model_pool, f"Model {model} not found in the model pool."
        return {param: self._draw(rng, values)
                for param, values in self.model_config.param_space(model).items()}

    def generate_random_team(self, rng: np.random.Generator, generation: int) -> Team:
        """A uniformly random allowed team with random blueprints; an unparented root."""
        allowed = self.topology.allowed_teams
        composition = allowed[int(rng.integers(len(allowed)))]
        roster = {role: self.generate_random_blueprint(rng) for role in sorted(composition)}
        return Team(roster=roster, parent_id0=-1, parent_id1=-1, generation=int(generation),
                    variation=None)

    # ------------------------------------------------------------ variation

    def variation_order(self, rng: np.random.Generator,
                        offspring_cnt: int) -> Tuple[List[str], int]:
        """Decide, per offspring, whether it comes from mutation ('m') or crossover ('c').

        Each offspring is a crossover with probability ``crossover_prob`` and a
        mutation otherwise. Returns the operator sequence and the total number of
        parents it consumes (mutation draws 1, crossover draws 2). Whether a
        crossover child is then mutated is decided later, in produce_offspring.
        """
        parent_count = {'m': 1, 'c': 2}

        order = rng.choice(['m', 'c'], offspring_cnt,
                           p=[1.0 - self.crossover_prob, self.crossover_prob])
        order = [str(op) for op in order]

        assert len(order) == offspring_cnt
        return order, int(sum(parent_count[op] for op in order))

    def produce_offspring(self,
                          rng: np.random.Generator,
                          offspring_cnt: int,
                          population: List[Team],
                          parent_ids: List[int],
                          order: List[str],
                          population_ids: List[int],
                          generation: int) -> List[Team]:
        """Generate offspring teams based on the given order of variation operations.

        A 'm' offspring is a mutation of a single parent. A 'c' offspring is a
        crossover of two parents, after which mutation is applied with probability
        ``1 - crossover_prob`` (and always applied when both parents are the same
        Team). The two are coupled on purpose to keep one knob: raising crossover_prob
        also lowers how often a crossover child is mutated, so more of them are exact
        recombinations (and more land on an already-archived genome).

        ``population_ids[i]`` is the archive id of ``population[i]``, so each
        offspring can be stamped with the archive ids of the parent(s) it was
        actually built from; ``generation`` is the generation these offspring
        belong to. A mutation records its single parent's id in both slots; a
        crossover records its two parents' ids, and the follow-up mutation (if any)
        keeps those two ids rather than pointing at the intermediate crossover
        child, which is never archived.

        Each offspring is labelled with the operators that ran on it: 'm' for a
        mutation, 'c' for a crossover left as is, 'cm' for a crossover followed by
        mutation. The label says which operators ran, not whether they changed
        anything.

        Every offspring is checked with ``validate_team`` before it is returned.
        """

        assert len(parent_ids) > 0
        assert len(population) > 0
        assert offspring_cnt > 0
        assert len(population_ids) == len(population), (
            f"population_ids ({len(population_ids)}) must line up with population "
            f"({len(population)}).")
        assert generation >= 0, f"Generation must be non-negative: {generation}."

        offspring = []

        p_id = 0
        for op in order:
            if op == 'm':
                parent_idx = parent_ids[p_id]
                parent_arch_id = int(population_ids[parent_idx])
                off = self.mutate(rng, population[parent_idx],
                                  parent_id0=parent_arch_id, parent_id1=parent_arch_id,
                                  generation=generation, variation='m')
                p_id += 1
            elif op == 'c':
                a_idx, b_idx = parent_ids[p_id], parent_ids[p_id+1]
                a_arch_id, b_arch_id = int(population_ids[a_idx]), int(population_ids[b_idx])
                off = self.crossover(rng, population[a_idx], population[b_idx],
                                     parent_id0=a_arch_id, parent_id1=b_arch_id,
                                     generation=generation, variation='c')

                if a_idx == b_idx or rng.random() < 1.0 - self.crossover_prob:
                    # Preserve the crossover's lineage through the follow-up mutation:
                    # the two real parents, not the throwaway crossover child.
                    off = self.mutate(rng, off,
                                      parent_id0=a_arch_id, parent_id1=b_arch_id,
                                      generation=generation, variation='cm')
                p_id += 2
            else:
                raise ValueError(f"Unknown operator: {op}")

            self.validate_team(off)
            offspring.append(off)

        assert len(offspring) == offspring_cnt
        assert p_id == len(parent_ids)

        return offspring

    def mutate(self, rng: np.random.Generator, parent: Team,
               parent_id0: int, parent_id1: int,
               generation: int, variation: str) -> Team:
        """Mutate a parent Team at the agent level and the topology level.

        The offspring's lineage is stamped from ``parent_id0``/``parent_id1`` and
        ``generation``. For a plain mutation both ids are the single parent's
        archive id; when this runs as the follow-up mutation after a crossover the
        caller passes the crossover's two parent ids through unchanged, so the
        recorded lineage stays the two grandparents rather than the throwaway
        crossover child. ``variation`` is 'm' for a plain mutation and 'cm' for the
        follow-up mutation. All four are required, so an offspring can never be
        recorded with no lineage by omission.

        Agent-level: every agent on the team rolls one of three outcomes:
            1. with probability ``prob_mut_blueprint``, get a whole new blueprint
               (a fresh model + a fresh parameter set),
            2. with probability ``prob_mut_parameters``, swap up to
               ``mut_parameters_cnt`` of its current model's parameters (clamped to
               how many parameters that model actually has), or
            3. otherwise keep the blueprint unchanged.

        Topology-level: with probability ``prob_mut_topology``, add or remove exactly
        one teammate. Only moves that land on an allowed team are offered, so a
        removal cannot leave a restricted team behind (and then be undone by the
        repair). A fair coin picks add vs remove when both are possible; if neither
        is, the mutation is skipped. An added worker gets a fresh random blueprint.
        """

        roster = parent.get_roster()

        threshold_blueprint = self.prob_mut_blueprint
        threshold_parameters = self.prob_mut_blueprint + self.prob_mut_parameters

        for role in sorted(roster):
            roll = rng.random()

            if roll < threshold_blueprint:
                # Outcome 1: a whole new blueprint (new model + fresh params).
                roster[role] = self.generate_random_blueprint(rng)

            elif roll < threshold_parameters:
                # Outcome 2: mutate a subset of the current model's parameters. Clamp
                # the count to the parameters the model has — a user asking for 2 when
                # only 1 exists would otherwise make rng.choice(replace=False) raise.
                # Single-value knobs (fixed settings such as Ollama's num_predict) are
                # skipped: redrawing one can never change it, so the mutation is wasted.
                blueprint = roster[role]
                model_space = self.model_config.param_space(blueprint.get_model())
                mutable = sorted(k for k, v in model_space.items() if len(v) > 1)
                n = min(self.mut_parameters_cnt, len(mutable))
                if n:
                    for param in rng.choice(mutable, n, replace=False):
                        blueprint.parameters[str(param)] = self._draw(rng, model_space[str(param)])

            # Outcome 3 (the remaining probability mass): keep the blueprint as is.

        # Topology-level mutation: add or remove exactly one teammate. The roster
        # keys ARE the team, so this operates directly on it.
        if rng.random() < self.prob_mut_topology:
            present = frozenset(roster)
            can_add = [w for w in sorted(self.topology.available_workers - present)
                       if self.topology.is_allowed(present | {w})]
            can_remove = [w for w in sorted(present - {SUPERVISOR})
                          if self.topology.is_allowed(present - {w})]

            if can_add and (not can_remove or rng.choice([True, False])):
                roster[self._draw(rng, can_add)] = self.generate_random_blueprint(rng)
            elif can_remove:
                roster.pop(self._draw(rng, can_remove))

        # Only an invalid parent reaches the repair with an invalid team: neither
        # step above can create one.
        roster = self._repair(rng, roster, donors=[])

        return Team(roster=roster,
                    parent_id0=int(parent_id0),
                    parent_id1=int(parent_id1),
                    generation=int(generation),
                    variation=variation)

    def crossover(self, rng: np.random.Generator, parent1: Team, parent2: Team,
                  parent_id0: int, parent_id1: int,
                  generation: int, variation: str) -> Team:
        """Crossover two parents into an offspring Team.

        Step through every teammate either parent fields. For each, flip a fair
        coin to pick which parent to inherit it from; if the chosen parent does not
        field that teammate, the offspring simply lacks it. The supervisor is on
        both parents, so exactly one supervisor always survives. The result may
        still be a composition that is not allowed; ``_repair`` then moves it to the
        nearest allowed team, drawing any role it adds back from the parents.

        ``parent_id0``/``parent_id1`` are the two parents' archive ids and are
        stamped onto the offspring as its lineage, alongside ``generation`` and
        ``variation`` ('c'; produce_offspring re-labels it 'cm' if it then mutates
        it). All four are required, so an offspring can never be recorded with no
        lineage by omission.
        """

        p1_roster = parent1.get_roster()
        p2_roster = parent2.get_roster()

        roster: Dict[str, BluePrint] = {}
        for role in sorted(set(p1_roster) | set(p2_roster)):
            source = p1_roster if rng.choice([True, False]) else p2_roster
            if role in source:
                roster[role] = source[role]

        roster = self._repair(rng, roster, donors=[p1_roster, p2_roster])

        return Team(roster=roster,
                    parent_id0=int(parent_id0),
                    parent_id1=int(parent_id1),
                    generation=int(generation),
                    variation=variation)

    def _repair(self, rng: np.random.Generator, roster: Dict[str, BluePrint],
                donors: List[Dict[str, BluePrint]]) -> Dict[str, BluePrint]:
        """``roster`` if its composition is allowed, else the nearest allowed team.

        Nearest means the fewest roles added or removed; ties are broken at random.
        On the chosen team, a role already in ``roster`` keeps its blueprint (with
        whatever mutation it received), a role a donor (a parent) fields takes that
        blueprint (a random donor's, if several do), and any other role gets a
        random blueprint.
        """
        present = frozenset(roster)
        if self.topology.is_allowed(present):
            return roster

        self.repair_count += 1
        allowed = self.topology.allowed_teams
        distance = [len(team ^ present) for team in allowed]
        nearest = [team for team, d in zip(allowed, distance) if d == min(distance)]
        target = nearest[int(rng.integers(len(nearest)))]

        repaired: Dict[str, BluePrint] = {}
        for role in sorted(target):
            if role in roster:
                repaired[role] = roster[role]
                continue
            holders = [donor for donor in donors if role in donor]
            repaired[role] = (self._draw(rng, holders)[role] if holders
                              else self.generate_random_blueprint(rng))
        return repaired
