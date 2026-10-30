"""The evolutionary algorithm (``EA``) that evolves agent teams with NSGA-II.

The two objectives are:
  1. Median CV ROC-AUC on the dev split (to be maximized)
  2. Cost of the team (to be minimized) -- USD when objective_2 is set to "cost_usd",
     otherwise the raw token count.

Variation operators (``reproduction.py``):
  1. Mutation: randomly alter an agent's parameters, or the team's roster
  2. Crossover: swap entire agents between two parents to create an offspring team

The startup checks are in ``preflight.py``.
"""

import numpy as np
import pandas as pd
import ray
from typeguard import typechecked
import os
import sys
from typing import Callable, List, Tuple, Dict, Any, Optional
import time
import uuid
import numpy.typing as npt

from ..Models.chat_model import check_param_names
from ..Agents.Tools.tool_resources import ToolResources
from ..Factories.base_factory import AgentFactory
from ..Factories.factory_set import split_factories
from ..Team_Builder.topology import build_topology_rules
from ..Models.model_config import ModelConfig
from ..Team_Builder.team import Team
from .reproduction import Reproduction
from .archive import Archive, SeenArchive
from ..Models.credentials import require_credentials
from ..Models.resources import ResourceAllocator, detect_resource_count
from ..Utils.log import warn
from ..Task.task import Task
from ..Task.data_split import describe_split, write_split_manifest
from ..Evaluation.evaluator import EvaluationRecorder, ProviderAbort
from ..Evaluation.result import EvalResult
from ..Evaluation.scoring import assert_enough_per_class, score_holdout
from ..Knowledge.catalog import assert_catalog_matches, load_catalog
from ..Knowledge.features import build_feature_frame
from ..Knowledge.knowledge_graph import KnowledgeGraph
from ..Knowledge.retriever import Retriever
from ..Knowledge.semantic_matcher import SemanticMatcher
from ..Team_Builder.graph import ModelBuilder, TeamGraphBuilder
from .objectives import OBJECTIVE_COST_USD, VALID_OBJECTIVES
from . import nsga_tool, preflight
from .backend import RayBackend
from .worker import RunContext, build_tools, store_dir
from ..Reporting.plots import plot_front
from ..Reporting.animations import ALL_FRONT_DIR, VISUALS_DIR, save_generation_frames
from ..Reporting.generation_summary import (GenerationMember, GenerationSummary, Quartiles,
                                            append_generation_members,
                                            append_generation_summary, print_generation_summary)
from ..Reporting.holdout import (HOLDOUT_RESULTS, HoldoutResult, print_holdout_summary,
                                 write_holdout_results)

@typechecked
class EA:
    def __init__(self,
                 seed: int,
                 pop_size: int,
                 factories: List[AgentFactory],
                 task_name: str,
                 target_column: str,
                 mut_parameters_cnt: int,
                 crossover_prob: float,
                 prob_mut_blueprint: float,
                 prob_mut_parameters: float,
                 prob_mut_topology: float,
                 cores: int,
                 eval_cnt: int,
                 csv_path: str,
                 catalog_path: str,
                 make_matcher: Callable[[], SemanticMatcher],
                 make_retriever: Optional[Callable[[str], Retriever]],
                 make_kg: Optional[Callable[[], KnowledgeGraph]],
                 save_directory: str,
                 holdout_fraction: float,
                 split_seed: int,
                 cv_folds: int,
                 model_config_path: str,
                 objective_2: str,
                 topology_rules: Dict[str, Any],
                 base_url: Optional[str] = None,
                 num_ctx: Optional[int] = None,
                 azure_deployment: Optional[str] = None,
                 api_version: Optional[str] = None,
                 archive_list: Optional[List[str]] = None,
                 build_model: Optional[ModelBuilder] = None,
                 stagger_seconds: float = 5.0) -> None:
        """
        Main class for the evolutionary algorithm.

        Parameters:

        seed: int
            Seed to initialize the random number generator.
        pop_size: int
            Population size.
        factories: List[AgentFactory]
            Every agent factory this run can field: at most one SupervisorFactory
            (leads teams with workers), at most one SingleAgentFactory (fills the
            Supervisor slot on a team with no workers), and one factory per worker
            role. The worker factories' role names are the roles teams are bred
            from, and each factory carries its role's prompts and data paths. Which
            of the two Supervisor-slot factories is required follows from
            topology_rules.
        topology_rules: Optional[Dict[str, Any]]
            required_workers / available_workers / restricted_teams; see
            ``Team_Builder.topology``. Name roles by a factory's ``role_name``.
        task_name: str
            Name of the task to evolve pipelines for.
        target_column: str
            Name of the target column in the dataset.
        mut_parameters_cnt: int
            Number of parameters to swap when an agent draws the "mutate
            parameters" outcome (clamped per model to what it exposes).
        prob_mut_topology: float
            Probability of a topology-level mutation, which adds or removes
            exactly one teammate (skipped if neither is possible).
        crossover_prob: float
            Probability that an offspring is produced by crossover; otherwise it
            is produced by mutation. A crossover child is then mutated with
            probability 1 - crossover_prob, so raising it also means fewer
            crossover children are mutated. 0.5 gives an even split.
        prob_mut_blueprint: float
            Per-agent probability of the "generate a whole new blueprint"
            mutation outcome (new model + fresh parameter set).
        prob_mut_parameters: float
            Per-agent probability of the "mutate mut_parameters_cnt parameters"
            outcome. prob_mut_blueprint + prob_mut_parameters must be <= 1; the
            remainder is the probability of keeping the agent unchanged.
        cores: int
            Number of Ray worker processes, and the most evaluation attempts in flight
            at once. ``cores=1`` evaluates one attempt at a time.
        csv_path: str
            Path to the cohort CSV. Read and split into dev / holdout once, here.
        catalog_path: str
            The feature catalog (``catalog.json``) the agents stage features from and
            the Evaluator encodes them with. Checked against the cohort here.
        make_matcher: Callable[[], SemanticMatcher]
            Builds the matcher behind the ``Semantic_Matcher`` tool, with no arguments.
            Must be built on ``catalog_path``; probed once here. Every worker process
            builds its own (``worker.py``), so pass a recipe, not a live object.
        make_retriever: Optional[Callable[[str], Retriever]]
            Builds the literature retriever behind the ``retrieve_context`` tool. Called
            with a folder the process may copy the store into. Required when a role in
            use retrieves (RAG, single agent); probed once here.
        make_kg: Optional[Callable[[], KnowledgeGraph]]
            Builds the biomedical graph and its node -> feature mapping, behind the
            ``kg_*`` tools, with no arguments. Required when a role in use searches it
            (KG, single agent); its mapped features must be in ``catalog_path``; probed
            once here.
        save_directory: str
            Directory to save results to.
        num_ctx: Optional[int]
            Ollama context window. Bounds prompt + completion together, so it must
            exceed the rendered system prompt plus the model's ``max_tokens`` knob. Ollama's own
            default is 4096, which silently truncates the supervisor prompt.
        holdout_fraction: float
            Proportion of the dataset reserved as the final test set. Split off
            before anything else runs and withheld from the agents for the whole
            search; scored once, at the end, on the Pareto front.
        split_seed: int
            Seed for the dev/holdout split. Deliberately separate from ``seed`` so
            that re-running the search with a different evolutionary seed still
            holds out the same patients and the two runs stay comparable.
        model_config_path: Optional[str]
            Path to the model_config.json describing each model's provider,
            capabilities and per-million-token rates. Required: a missing file
            fails rather than pricing every model at zero.
        objective_2: str
            What NSGA-II minimizes alongside performance: "cost_usd" (each agent
            priced at its own model's rates) or "token_count" (raw tokens). See
            ``objectives.py``. Required: the two produce different fronts.
        azure_deployment: Optional[str]
            Default Azure deployment name for models with no config entry.
        api_version: Optional[str]
            Default Azure API version for models with no config entry.
        cv_folds: int
            Stratified folds used to score each pipeline within the dev split.
        archive_list: Optional[List[str]]
            Paths to archive.json files written by *other* runs/replicates. Any
            agent team already evaluated in one of them has its stored scores
            replayed instead of being re-evaluated here, so sequential replicates
            never pay to measure the same team twice. This includes the initial
            population, which may therefore start with teams first seen elsewhere.
            The replayed teams are archived into this run's own archive.json too, so
            each run stays self-contained and a later replicate need only point at
            the prior runs' archives. Omit (or pass an empty list) to disable.
        build_model: Optional[ModelBuilder]
            Builds each agent's chat model from its blueprint; ``build_chat_model``
            by default. Replaced in tests; a replacement must pickle, as it is sent to
            the workers.
        stagger_seconds: float
            Minimum gap between launching one evaluation attempt and the next, so the
            teams do not all start their LLM calls at once and exhaust the provider's
            per-minute token limit.

        """
        assert 0 < pop_size, f"Population size must be greater than 0: {pop_size}."
        assert 0 < cores, f"Number of cores must be greater than 0: {cores}."
        assert os.path.isdir(save_directory), f"Save directory does not exist: {save_directory}."
        assert len(task_name) > 0, f"Task name cannot be empty: {task_name}."
        assert len(target_column) > 0, f"Target column cannot be empty: {target_column}."
        assert eval_cnt > 0, f"Number of evaluations must be greater than 0: {eval_cnt}."
        assert 0.0 < holdout_fraction < 1.0, f"Holdout fraction must be strictly between 0 and 1: {holdout_fraction}."
        assert cv_folds >= 2, f"Cross-validation needs at least 2 folds: {cv_folds}."
        assert stagger_seconds >= 0.0, f"stagger_seconds must be non-negative: {stagger_seconds}."

        self.seed = seed
        self.pop_size = pop_size
        self.cores = cores
        self.save_directory = save_directory
        self.task_name = task_name
        self.target_column = target_column
        self.holdout_fraction = holdout_fraction
        self.split_seed = split_seed
        self.cv_folds = cv_folds
        self.stagger_seconds = float(stagger_seconds)

        # The task is built once and shared by every later reader, so the CSV is
        # read and split a single time. A missing file or target column fails
        # here, before any model is called.
        self.task = Task(task_name=task_name, csv_path=csv_path,
                         target_column=target_column,
                         holdout_fraction=holdout_fraction, split_seed=split_seed)
        # Every team is scored with cv_folds-fold stratified CV on the dev split, so
        # a dev split too small for that fails now rather than at the first score.
        assert_enough_per_class(self.task.get_dataframe()[target_column].to_numpy(),
                                cv_folds)
        # Every catalog feature must be encodable from the cohort (dev and holdout), so a
        # catalog built from other data fails now, not when a team commits a feature.
        self.catalog = load_catalog(catalog_path)
        assert_catalog_matches(self.catalog, pd.concat([self.task.get_dataframe(),
                                                        self.task.get_holdout_dataframe()]))
        # Everything this run writes goes under {save_directory}/runs/{task_name}.
        self.run_dir = os.path.join(save_directory, "runs", task_name)
        # The run's tools, built here for the probes and for check_models. Each worker
        # process builds its own from the same recipes (worker.py).
        matcher, retriever, kg = build_tools(make_matcher, make_retriever, make_kg,
                                             store_dir(self.run_dir, "driver"))
        # The statistics tools analyse the dev split only, the rows the CV folds come from.
        # Checked before the probes, so a tool of the wrong type fails as such.
        resources = ToolResources(catalog=self.catalog, matcher=matcher,
                                  dataset=self.task.get_dataframe(),
                                  target_column=target_column, retriever=retriever, kg=kg)
        preflight.probe_matcher(matcher, self.catalog, catalog_path)

        # remote Ollama server settings (Ollama agent runtime, wired when agents.py lands)
        self.base_url = base_url
        self.num_ctx = num_ctx
        # The run's model table: provider, per-model param_space, and pricing.
        # Constructed here and held for the run -- there is no process-wide
        # singleton -- so Reproduction below can read the model pool out of it.
        # NOTE: base_url / num_ctx configure the Ollama *agent* runtime, which lives
        # in agents.py and is not yet refactored into Source/; wire that here when it
        # lands. They are already stored on self above.
        self.model_config = ModelConfig(model_config_path,
                                        azure_deployment=azure_deployment,
                                        api_version=api_version)
        # Objective 2 is dollars or tokens; the EA owns this choice, not the config.
        self.objective_2 = str(objective_2)
        if self.objective_2 not in VALID_OBJECTIVES:
            raise ValueError(
                f"objective_2 must be one of {VALID_OBJECTIVES}, got '{self.objective_2}'")
        # With cost_usd but nothing priced, objective 2 is constant 0.0 and NSGA-II
        # collapses to single-objective.
        if self.objective_2 == OBJECTIVE_COST_USD and not self.model_config.has_priced_models():
            warn("objective_2 is 'cost_usd' but no model in the model pool has a "
                 "non-zero rate. Objective 2 will be constant 0.0 and NSGA-II "
                 "collapses to single-objective. Fill in the pricing.")
        # Offline, before anything is spent: every param_space knob must be one its
        # model's client recognizes. Whether the API accepts the values is check_models.
        check_param_names(self.model_config)
        self.eval_cnt = eval_cnt

        self.recursion_limit = self.model_config.recursion_limit
        # The factories decide which roles exist; topology_rules decides which
        # teams of them are allowed. Checked together, so a run never breeds a
        # team it cannot build.
        self.factories = split_factories(factories)
        self.topology = build_topology_rules(topology_rules, self.factories.workers)
        preflight.check_factories_cover_topology(self.factories, self.topology)

        # Reproduction draws models and parameter values from the model config and
        # compositions from the topology; it is the only place teams are made.
        self.reproduction = Reproduction(topology=self.topology,
                                         model_config=self.model_config,
                                         crossover_prob=crossover_prob,
                                         prob_mut_blueprint=prob_mut_blueprint,
                                         prob_mut_parameters=prob_mut_parameters,
                                         mut_parameters_cnt=mut_parameters_cnt,
                                         prob_mut_topology=prob_mut_topology)

        # rng to be used for all random operations within the evolver to ensure reproducibility
        self.rng: np.random.Generator = np.random.default_rng(seed)

        # the population of pipelines to be evolved
        self.population: List[Team] = []

        # The archive of every unique team genome evaluated this run. It dedups
        # by genome, hands each new team a sequential integer id (which is what
        # offspring lineage is recorded against), and stores the full evaluation
        # record so a re-created genome need never be re-evaluated.
        self.archive: Archive = Archive(recursion_limit=self.recursion_limit)

        # Read-only history from other runs: teams these replicates already
        # evaluated are replayed rather than re-evaluated here (see collect_
        # teams_to_evaluate).
        archive_list = archive_list or []
        self.seen_archive: SeenArchive = SeenArchive.from_paths(
            archive_list, recursion_limit=self.recursion_limit)
        if len(self.seen_archive):
            print(f"Loaded {len(self.seen_archive)} previously-evaluated teams from "
                  f"{len(archive_list)} external archive(s); matching candidates will "
                  f"be replayed instead of re-evaluated.", flush=True)

        # current generation index (-1 = initial population, 0 = first round of
        # selection and offspring); used to namespace per-run history files.
        self.current_gen: int = -1
        # Digits generation numbers are zero-padded to in the GIF frame names, so the
        # files sort in generation order; evolve() sets it from the last generation index.
        self.gen_width: int = 1

        # Filled by self.evaluation() so the caller can report timings without
        # threading them back through the return value: the whole call's wall
        # time, and each evaluated team's time across its eval_cnt evaluations.
        self.last_eval_seconds: float = 0.0
        self.last_team_seconds: List[float] = []
        # Filled by evaluation: the most attempts in flight at once during the last batch.
        self.last_peak_in_flight: int = 0
        # Filled by collect_teams_to_evaluate: how many of the last batch were
        # replayed from this run's archive and from external archives.
        self.last_replayed_run: int = 0
        self.last_replayed_external: int = 0

        # Every plot, GIF and the generation CSV go under the run folder; the plots
        # and GIFs under its Visuals/ folder.
        self.front_png_dir = os.path.join(self.run_dir, VISUALS_DIR, ALL_FRONT_DIR)
        self.generation_csv = os.path.join(self.run_dir, "generation_summary.csv")
        # Each generation's candidates by archive id: origin, parents, front, outcome.
        self.members_csv = os.path.join(self.run_dir, "generation_members.csv")
        # Every genome evaluated so far, rewritten after each generation.
        self.archive_path = os.path.join(self.run_dir, "archive.json")
        # The final population's Pareto front scored on the holdout, written once at the end.
        self.holdout_path = os.path.join(self.run_dir, HOLDOUT_RESULTS)

        # Azure resources (R1_.., R2_.. in the env) to spread evaluations across;
        # a gap in the numbering raises. ModelConfig above has already loaded .env.
        # Each in-flight evaluation is pinned to one of them (see evaluation), so no
        # (resource, model) deployment is shared.
        self.n_resources = detect_resource_count()
        print(f"Resource pool: {self.n_resources} Azure resource(s)"
              + (f" (R1..R{self.n_resources})" if self.n_resources else ""), flush=True)

        # Fail on a missing API key here. Only the providers the model pool can actually
        # reach are checked, so an unused provider in the config never blocks a local run.
        require_credentials(self.model_config.providers_in_use(
            self.reproduction.model_pool))

        # Only factories some allowed team is built from are checked, so an unused role
        # never blocks a run.
        in_use = preflight.factories_in_use(self.factories, self.topology)
        preflight.probe_retriever(retriever, self.catalog, in_use)
        preflight.probe_kg(kg, self.catalog, in_use)
        # Fail fast if a role's external service is unreachable.
        for factory in in_use:
            factory.preflight()

        # Built once, for check_models: each worker builds its own tools (worker.py).
        build_graph = TeamGraphBuilder(self.factories, resources, self.task,
                                       self.model_config, resource=1,
                                       build_model=build_model)
        # The same builder the teams use, for check_models.
        self.build_model: ModelBuilder = build_graph.build_model
        # The run's record of every evaluation: the CSV, and the provider-abort count (too
        # many evaluations in a row lost to the provider raise ProviderAbort).
        self.recorder = EvaluationRecorder(
            self.run_dir, abort_after=self.model_config.abort_after_consecutive_infra_failures)
        # Ray runs the evaluation attempts, one per CPU of its worker processes. It starts
        # here, after ModelConfig has loaded .env, so the workers inherit the credentials.
        # The workers import this run's modules from the driver's own import path. The
        # dashboard is served while the run lasts; Ray prints its URL here.
        ray.init(num_cpus=cores, ignore_reinit_error=True, include_dashboard=True,
                 runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(sys.path)}})
        self.run_context = RunContext(
            run_key=uuid.uuid4().hex, run_dir=self.run_dir, task=self.task,
            catalog=self.catalog, factories=self.factories, model_config=self.model_config,
            cv_folds=cv_folds, seed=seed, make_matcher=make_matcher,
            make_retriever=make_retriever, make_kg=make_kg, build_model=build_model)
        self.backend = RayBackend(self.run_context)
        print(flush=True)

    def check_models(self, models: List[str]) -> None:
        """The live hello-world check of ``models`` (``preflight.check_models``), with the
        builder the teams use. Run before ``evolve`` (runner.py ``--check_models``)."""
        preflight.check_models(models, self.model_config, self.build_model)

    def log_data_split(self) -> None:
        """Announce the dev/holdout split and write its manifest.

        Called once, before the first team runs, so the run log opens with the row
        counts and class balance on each side. The manifest lists the holdout row indices.
        """
        dev, holdout = self.task.get_dataframe(), self.task.get_holdout_dataframe()
        print(f"\n{'=' * 80}", flush=True)
        print(f"  DATA SPLIT (seed {self.split_seed}, "
              f"holdout {self.holdout_fraction:.0%})", flush=True)
        print(f"  {describe_split(dev, holdout, self.target_column)}", flush=True)
        print(f"  team scores are {self.cv_folds}-fold cross-validated within "
              f"the dev split", flush=True)
        os.makedirs(self.run_dir, exist_ok=True)
        path = write_split_manifest(
            out_dir=self.run_dir, dev=dev, holdout=holdout,
            target_column=self.target_column, holdout_fraction=self.holdout_fraction,
            seed=self.split_seed, csv_path=self.task.csv_path)
        print(f"  split manifest      : {path}", flush=True)
        print(f"{'=' * 80}\n", flush=True)

    # -------------------------------- evolver -----------------------------------------

    def evolve(self, gens: int) -> None:
        """
        Evolve the population of teams for a given number of generations.

        NSGA-II algorithm:
            1. Initialize and evaluate the population (generation -1).
            2. For generations 0 .. gens-1:
                a. Select parents.
                b. Generate offspring through crossover and mutation,
                   dropping duplicates of each other or of the population.
                c. Evaluate offspring (replaying any genome already measured).
                d. Select survivors from the current population and offspring.
                e. Save the archive.
                f. Report the generation (members rows, GIF frames, summary).
            3. Plot every team the run evaluated (``all_front.png``).
            4. Score the final population's Pareto front on the holdout.

        Args:
            gens: Number of generations to evolve the population.
        """

        assert gens > 0, f"Number of generations must be greater than 0: {gens}."

        # Pad generation numbers to the last index: gens=10 -> gen0..gen9,
        # gens=100 -> gen00..gen99.
        self.gen_width = len(str(gens - 1))

        # step 0: split the data before running, and record it.
        self.log_data_split()

        # step 1: initialize the population. It saves the archive before the check
        # below, so a run whose initial teams all failed still keeps what it paid for.
        self.initialize_population()
        assert self.population, (
            "Every initial team failed; there is nothing to evolve. Check the evaluation "
            "logs before rerunning.")

        # step 2: evolve for gens generations (0 .. gens-1)
        print(f"Starting evolution for {gens} generations...", flush=True)
        for gen in range(gens):
            # Generation -1 is the initial population; generation 0 is the first
            # round of offspring.
            self.current_gen = gen
            print(f"Generation {gen} ({gen + 1}/{gens})", flush=True)
            gen_start = time.time()

            # step 2a: decide the variation order and select parents
            time_start = time.time()
            order, parent_cnt = self.reproduction.variation_order(self.rng, self.pop_size)
            parent_ids = self.parent_selection(parent_cnt)
            parent_seconds = time.time() - time_start

            # step 2b: generate offspring through crossover and mutation.
            # Every population member is archived, so each resolves to the id
            # produce_offspring stamps on its offspring as lineage.
            time_start = time.time()
            population_ids = [self.archive.get_id(team) for team in self.population]
            offspring = self.reproduction.produce_offspring(
                rng=self.rng,
                offspring_cnt=self.pop_size,
                population=self.population,
                parent_ids=parent_ids,
                order=order,
                population_ids=population_ids,
                generation=gen)
            offspring = self.deduplicate_offspring(offspring)
            offspring_seconds = time.time() - time_start

            # step 2c: evaluate offspring. A genome measured earlier this run or in an
            # external archive has its record replayed instead.
            teams_to_evaluate = self.collect_teams_to_evaluate(offspring)
            self.evaluation(teams_to_evaluate)

            # Archive after evaluation so every record is complete.
            self._record_teams(offspring)

            # Every candidate this generation, failures included, before truncation.
            candidates = self.population + offspring

            # step 2d: select survivors from the population and the offspring that
            # scored. The population is carried forward already viable.
            time_start = time.time()
            self.population = self.survival_selection(
                self.population + self.exclude_failed_teams(offspring))
            survival_seconds = time.time() - time_start

            # step 2e: save the archive, so a crash loses at most this generation. Saved
            # before the report, so every id in the members rows is in archive.json.
            self.save_archive()

            # step 2f: report the generation (members rows, GIF frames, summary)
            self.report_generation(generation=gen,
                                   gen_seconds=time.time() - gen_start,
                                   stage_times={"parent selection": parent_seconds,
                                                "offspring": offspring_seconds,
                                                "survival": survival_seconds},
                                   candidates=candidates)

        # step 3: one front over every team the run evaluated, across all generations
        self._save_archive_front()

        # step 4: the search is over, so the holdout is finally used, once
        self.holdout_evaluation()
        return

    def initialize_population(self) -> None:
        """
        Function to initialize the population of pipelines for self.population.
        Size of self.population must be self.pop_size.
        """

        print(f"Initializing population...", flush=True)
        start_time = time.time()

        # Fill the population with distinct random teams
        max_attempts = 100 * self.pop_size
        seen_keys = set()
        attempts = 0
        while len(self.population) < self.pop_size and attempts < max_attempts:
            attempts += 1
            # Initial-population teams have no parents and belong to generation -1;
            # generation 0 is the first round of selection and offspring.
            team = self.reproduction.generate_random_team(rng=self.rng, generation=-1)
            key = self.archive.key(team)
            if key not in seen_keys:
                seen_keys.add(key)
                self.population.append(team)
        assert len(self.population) == self.pop_size, (
            f"Found only {len(self.population)} distinct teams in {max_attempts} attempts; "
            f"pop_size is {self.pop_size}. Lower pop_size or widen the param_space/topology.")
        if attempts > self.pop_size:
            print(f"  Drew {attempts} random teams to get {self.pop_size} distinct ones.", flush=True)

        # evaluate the initialized population. Any genome an external archive
        # already measured (collect_teams_to_evaluate) has its record replayed.
        full_population = self.population
        teams_to_evaluate = self.collect_teams_to_evaluate(full_population)
        self.evaluation(teams_to_evaluate)

        # Record every genome from the initial population
        self._record_teams(full_population)

        # Only pipelines not failed are carried forward
        self.population = self.exclude_failed_teams(full_population)

        print(f"Population initialized and evaluated in {time.time() - start_time:.2f} seconds.", flush=True)

        # Saved before the report, so every id in the members rows is in archive.json.
        self.save_archive()

        # Generation -1 is the initial population
        self.report_generation(generation=-1, gen_seconds=time.time() - start_time,
                               stage_times={}, candidates=full_population)
        return

    def holdout_evaluation(self) -> List[HoldoutResult]:
        """Score the final population's Pareto front on the holdout, once, after evolution.

        No LLM is called. Each front team's median-evaluation features
        (``Team.get_median_features``) are encoded on dev and on the holdout, both
        fitted on dev, and one XGBoost model fitted on all of dev is scored on the
        holdout. Nothing feeds back into selection, so the holdout stays an estimate of
        performance on unseen rows. Teams run one at a time, cheapest first; the
        results go to ``holdout_results.json`` and the console.

        Every front team scored above 0, so it committed features that were already
        encoded and scored on dev; a failure here is a bug. It is printed with the
        team's archive id and raised. The archive is saved before this runs.
        """
        dev, holdout = self.task.get_dataframe(), self.task.get_holdout_dataframe()
        teams, plot = self._front_plot_data(self.population)
        assert plot["front"].size, "the final population has no viable team to score"
        print(f"Scoring the {plot['front'].size}-team Pareto front on the holdout...",
              flush=True)

        results: List[HoldoutResult] = []
        for i in plot["front"]:
            team, team_id = teams[i], plot["ids"][i]
            features = team.get_median_features()
            try:
                names = [record["feature_name"] for record in features]
                dev_frame = build_feature_frame(dev, names, self.catalog, self.target_column,
                                                fit_df=dev)
                holdout_frame = build_feature_frame(holdout, names, self.catalog,
                                                    self.target_column, fit_df=dev)
                score = score_holdout(dev_frame, holdout_frame, self.target_column, self.seed)
            except Exception as exc:
                print(f"  [holdout] team {team_id} failed: {type(exc).__name__}: {exc}",
                      flush=True)
                raise
            results.append(HoldoutResult(team_id=team_id,
                                         evaluation=team.median_evaluation(),
                                         median_roc_auc=team.get_median_performance(),
                                         median_token_count=team.get_median_token_count(),
                                         median_cost_usd=team.get_median_cost(),
                                         dev_metrics=team.get_median_metrics(),
                                         features=features,
                                         score=score))

        write_holdout_results(self.holdout_path, task_name=self.task_name,
                              objective_2=self.objective_2, seed=self.seed,
                              n_dev_rows=len(dev), n_holdout_rows=len(holdout),
                              results=results)
        print_holdout_summary(results, objective_2=self.objective_2,
                              n_holdout_rows=len(holdout), path=self.holdout_path)
        return results

    # -------------------------------- objectives -----------------------------------------

    def objective_2_of(self, team: Team) -> float:
        """The value NSGA-II minimizes alongside performance.

        The median across the team's evaluations, matching the median performance
        NSGA-II ranks on, so both objectives describe its typical run. USD when the
        run optimizes cost, otherwise the token count.
        """
        if self.objective_2 == OBJECTIVE_COST_USD:
            return float(team.get_median_cost())
        return float(team.get_median_token_count())

    def objective_2_values(self, teams: List[Team]) -> List[float]:
        return [self.objective_2_of(team) for team in teams]

    def get_team_scores(self, teams: List[Team]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """(median performance, objective 2) for each team, by position.

        Both as measured: nsga_tool maximizes the first and minimizes the second
        itself, so nothing is negated or weighted here.
        """
        return (np.array([team.get_median_performance() for team in teams], dtype=np.float64),
                np.array(self.objective_2_values(teams), dtype=np.float64))

    @staticmethod
    def viable_positions(teams: List[Team]) -> List[int]:
        """Positions of the teams that actually produced a score.

        A median performance of 0.0 is the sentinel every failure path writes (the
        Supervisor never returned a feature set, it was empty, or XGBoost errored).
        Selection and reporting use this one rule.
        """
        return [i for i, team in enumerate(teams) if team.get_median_performance() > 0.0]

    def exclude_failed_teams(self, teams: List[Team]) -> List[Team]:
        """The teams that scored (see ``viable_positions``), in their original order."""
        return [teams[i] for i in self.viable_positions(teams)]

    # -------------------------------- selection -----------------------------------------

    def parent_selection(self, parent_cnt: int) -> List[int]:
        """Positions in ``self.population`` of ``parent_cnt`` parents, by binary tournament.

        Each tournament draws two teams and keeps the lower NSGA-II rank, then the larger crowding
        distance. A population of one has no tournament to run: that member is every parent.
        """
        assert parent_cnt > 0, f"parent_cnt must be positive: {parent_cnt}."
        n = len(self.population)
        assert n > 0, "Population is empty; nothing to select parents from."
        assert len(self.viable_positions(self.population)) == n, (
            "Population holds a failed team; only teams that scored are carried forward.")
        if n == 1:
            return [0] * parent_cnt

        performances, objective2 = self.get_team_scores(self.population)
        ranks = nsga_tool.non_dominated_sorting(performances, objective2)
        distances = nsga_tool.crowding_distance(performances, objective2, ranks)
        parent_ids = []
        for _ in range(parent_cnt):
            t1, t2 = self.rng.choice(n, size=2, replace=False)
            parent_ids.append(nsga_tool.nds_tournament(ranks, distances, int(t1), int(t2)))
        return parent_ids

    def deduplicate_offspring(self, offspring: List[Team]) -> List[Team]:
        """``offspring`` without teams whose genome is already in this generation.

        A genome is dropped when it is a clone in the offspring set or ``self.population``:
        keeping it would spend an evaluation (or a replay) and let one genome take two survival
        slots. The first offspring of each clone is kept. The batch may shrink and is not refilled;
        the population is carried forward alongside it, so survival selection always has candidates.

        A genome archived earlier in the run but no longer in the population is
        kept; collect_teams_to_evaluate replays its record.
        """
        population_keys = {self.archive.key(team) for team in self.population}
        # Initialization draws distinct teams, this dedups offspring against the
        # population, and survival keeps a subset of the two: a repeat means one broke.
        assert len(population_keys) == len(self.population), (
            "Population holds a duplicated genome; it should be distinct by construction.")
        seen = set(population_keys)
        unique: List[Team] = []
        in_batch = in_population = 0
        for team in offspring:
            key = self.archive.key(team)
            if key in seen:
                if key in population_keys:
                    in_population += 1
                else:
                    in_batch += 1
                continue
            seen.add(key)
            unique.append(team)
        if in_batch or in_population:
            print(f"  [offspring] dropped {in_batch + in_population} duplicate offspring "
                  f"({in_batch} within the batch, {in_population} already in the population); "
                  f"{len(unique)}/{len(offspring)} remain.", flush=True)
        return unique

    def survival_selection(self, candidates: List[Team]) -> List[Team]:
        """The next population: up to ``pop_size`` of ``candidates`` by NSGA-II.

        ``candidates`` are the population and the offspring that scored. When more
        than ``pop_size`` remain, whole fronts are kept in rank order and the front
        that does not fit is split by crowding distance (ties broken with self.rng).
        """
        assert candidates, "No candidates to select survivors from."
        assert len(self.viable_positions(candidates)) == len(candidates), (
            "Candidates hold a failed team; exclude failures before survival selection.")
        keys = {self.archive.key(team) for team in candidates}
        assert len(keys) == len(candidates), (
            "Candidates hold a duplicated genome; offspring are deduplicated against "
            "the population before survival selection.")

        if len(candidates) <= self.pop_size:
            if len(candidates) < self.pop_size:
                print(f"  [selection] only {len(candidates)} candidates scored; population "
                      f"shrinks to {len(candidates)} (pop_size={self.pop_size}) rather "
                      f"than carrying failures.", flush=True)
            return list(candidates)

        performances, objective2 = self.get_team_scores(candidates)
        ranks = nsga_tool.non_dominated_sorting(performances, objective2)
        distances = nsga_tool.crowding_distance(performances, objective2, ranks)
        survivor_ids = nsga_tool.nds_truncate(ranks, distances, self.pop_size, self.rng)
        assert len(survivor_ids) == self.pop_size, (
            f"nds_truncate kept {len(survivor_ids)} survivors, expected {self.pop_size}.")
        return [candidates[i] for i in survivor_ids]

    # -------------------------------- evaluation -----------------------------------------

    def evaluation(self, teams: List[Team]) -> None:
        """Evaluate each team ``eval_cnt`` times through the backend and archive it.

        Every team gets its id up front, since its results folder (``Team_Results/Team_{id}``)
        is named after it, and its ``eval_cnt`` attempts are queued smallest team first. At
        most ``cores`` attempts are in flight. Each is pinned to one Azure resource (resource 1
        when none are configured), so no two in flight share a (resource, model) deployment's
        token quota, and launches are at least ``stagger_seconds`` apart. A team's record is
        attached once its last attempt lands. ``teams`` must all be new genomes
        (``collect_teams_to_evaluate``).

        ``ProviderAbort`` (too many evaluations in a row lost to the provider) cancels the
        attempts still running and saves the archive, so every team finished so far can be
        replayed by a rerun through ``archive_list``, then re-raises. A team not yet finished
        is saved without results and is evaluated again on a rerun. Any other error cancels
        the attempts still running and propagates: it is a bug, and it stops the run.
        """
        start_time = time.time()
        for team in teams:
            assert not self.archive.contains(team), (
                f"Team {self.archive.get_id(team)} is already archived; replay its "
                f"record (collect_teams_to_evaluate) instead of evaluating it again.")
        team_ids = [self.archive.reserve(team) for team in teams]

        team_sizes = [len(team.get_roster()) for team in teams]
        model_sets = [{str(bp.get_model()) for bp in team.get_roster().values()}
                      for team in teams]
        for pos, models in enumerate(model_sets):
            assert models, f"team at position {pos} has an empty model set"
        # Smallest teams first: they reserve the fewest (resource, model) pairs, so they pack
        # densely and drain first. Ties keep the batch order.
        order = sorted(range(len(teams)),
                       key=lambda pos: (team_sizes[pos], len(model_sets[pos]), pos))
        work: List[Tuple[int, int]] = [(pos, e) for pos in order for e in range(self.eval_cnt)]
        allocator = ResourceAllocator(self.n_resources) if self.n_resources else None

        results: List[Dict[int, EvalResult]] = [dict() for _ in teams]
        seconds = [0.0] * len(teams)
        # handle -> (team position, eval index, resource id)
        inflight: Dict[Any, Tuple[int, int, int]] = {}
        next_launch = 0.0
        peak = 0

        def launch_ready() -> None:
            """Launch queued attempts while the window has room, smallest team first. An
            attempt no resource can host right now is skipped, so it does not block the
            queue behind it."""
            nonlocal next_launch, peak
            i = 0
            while len(inflight) < self.cores and i < len(work):
                if time.monotonic() < next_launch:
                    break
                pos, eval_index = work[i]
                resource_id = 1
                if allocator is not None:
                    assigned = allocator.try_assign(model_sets[pos])
                    if assigned is None:
                        i += 1
                        continue
                    resource_id = assigned
                del work[i]  # the list shifts, so do not advance i
                handle = self.backend.submit(teams[pos], team_ids[pos], eval_index, resource_id)
                inflight[handle] = (pos, eval_index, resource_id)
                next_launch = time.monotonic() + self.stagger_seconds
                peak = max(peak, len(inflight))
            return

        try:
            launch_ready()
            while inflight or work:
                if not inflight:
                    # Nothing is running, so only the stagger holds the next launch.
                    time.sleep(max(0.0, next_launch - time.monotonic()))
                    launch_ready()
                    assert inflight, "nothing in flight and nothing launchable"
                    continue
                timeout = None
                if work and len(inflight) < self.cores and time.monotonic() < next_launch:
                    # The stagger is what holds the next launch: wake up when it ends.
                    timeout = max(0.0, next_launch - time.monotonic())
                finished = self.backend.wait(list(inflight), timeout)
                if finished is None:
                    launch_ready()
                    continue
                handle, result = finished
                pos, eval_index, resource_id = inflight.pop(handle)
                if allocator is not None:
                    allocator.release(resource_id, model_sets[pos])
                results[pos][eval_index] = result
                seconds[pos] += result.eval_seconds
                self.recorder.record(result)
                if len(results[pos]) == self.eval_cnt:
                    self._archive_team(teams[pos], results[pos])
                launch_ready()
        except BaseException as exc:
            self.backend.cancel(list(inflight))
            if isinstance(exc, ProviderAbort):
                self.save_archive()
            raise

        assert not work, f"{len(work)} attempts were never launched"
        assert allocator is None or allocator.busy_pairs() == 0, (
            f"allocator leaked {allocator.busy_pairs()} busy (resource, model) pairs")
        self.last_eval_seconds = time.time() - start_time
        self.last_team_seconds = seconds
        self.last_peak_in_flight = peak
        return

    def _archive_team(self, team: Team, results: Dict[int, Any]) -> None:
        """Attach a team's ``eval_cnt`` results (``EvalResult``s, by evaluation index) in
        evaluation order, so the latest is the highest-indexed attempt, then finalize its
        archive record."""
        for eval_index in range(self.eval_cnt):
            team.set_evaluation_result(**results[eval_index].team_kwargs())
        self.archive.finalize(team)
        return

    # ------------------------------- archive -----------------------------------------

    def collect_teams_to_evaluate(self, teams: List[Team]) -> List[Team]:
        """
        Return the teams whose genomes still need evaluating
        and replay the evaluation record for any already-seen genome.
        """
        to_evaluate: List[Team] = []
        reused_run = 0
        reused_seen = 0
        # External archives keep the eval_cnt of the run that wrote them.
        mismatched = 0
        for team in teams:
            if self.archive.contains(team):
                # Seen this genome earlier this run: replay
                self.archive.apply_results(team)
                reused_run += 1
            elif self.seen_archive.contains(team):
                # Seen this genome in another replicate's archive: replay
                self.seen_archive.apply_results(team)
                reused_seen += 1
                mismatched += len(team.get_all_performance()) != self.eval_cnt
            else:
                to_evaluate.append(team)
        if reused_run or reused_seen:
            print(f"  [archive] reusing {reused_run + reused_seen}/{len(teams)} "
                  f"teams ({reused_run} from this run, {reused_seen} from external "
                  f"archives); evaluating the remaining {len(to_evaluate)} new genomes.",
                  flush=True)
        if mismatched:
            warn(f"{mismatched} team(s) replayed from external archives carry a different "
                 f"number of evaluations than this run's eval_cnt ({self.eval_cnt}); the "
                 f"generation summary's every-evaluation totals are not comparable across "
                 f"generations while they are in the candidates.", source="archive")
        self.last_replayed_run = reused_run
        self.last_replayed_external = reused_seen
        return to_evaluate

    def _record_teams(self, teams: List[Team]) -> None:
        """Add every not-yet-seen genome in ``teams`` to the archive.

        Only called after evaluation, which has already archived the teams it
        evaluated. A genome still missing was replayed from an external archive by
        collect_teams_to_evaluate, and is recorded with that file and id as source.
        """
        for team in teams:
            if not self.archive.contains(team):
                source = (self.seen_archive.source_of(team)
                          if self.seen_archive.contains(team) else None)
                self.archive.add(team, source=source)
        return

    # ------------------------------- data tracking -----------------------------------------

    def report_generation(self, generation: int, gen_seconds: float,
                          stage_times: Dict[str, float], candidates: List[Team]) -> None:
        """Record each candidate's origin, front and outcome, save the generation's
        GIF frames, print the generation summary and append its CSV row.

        ``candidates`` are every team that competed this generation (the initial
        population, or the population plus offspring before truncation); the summary
        describes them, so it is a snapshot of the whole generation. The frames are
        drawn from the run folder (``archive.json``, saved before this call, and the
        members rows just appended), scaled to the run so far; the GIFs, and every frame
        redrawn on the whole run's scale, are made once the run ends (runner.py).
        Errors are caught and printed here: reporting must never take
        down a run that has already spent hours of LLM time.
        """
        # ---- membership rows, on their own so a failed plot cannot skip them
        try:
            append_generation_members(generation, self._generation_members(generation, candidates),
                                      self.members_csv)
        except Exception as exc:
            print(f"  [reporting] generation {generation} members failed: {exc!r}", flush=True)

        # ---- GIF frames, on their own so a failed plot cannot skip the summary
        plot_path = self._save_generation_frames(generation)

        try:
            # ---- candidate snapshot (distinct genomes, as in the plots)
            teams, _ = self._front_plot_data(candidates)
            viable = set(self.viable_positions(teams))
            successful = [t for i, t in enumerate(teams) if i in viable]
            failed = [t for i, t in enumerate(teams) if i not in viable]

            def total(group: List[Team], values: str) -> float:
                return float(sum(sum(getattr(t, values)()) for t in group))

            total_tokens = int(total(teams, "get_all_token_count"))
            total_cost = total(teams, "get_all_cost")
            failed_tokens = int(total(failed, "get_all_token_count"))
            failed_cost = total(failed, "get_all_cost")

            n_new = len(self.last_team_seconds)
            n_replayed = self.last_replayed_run + self.last_replayed_external
            n_carried = len(teams) - n_new - n_replayed
            assert n_carried >= 0, (
                f"{len(teams)} distinct candidates but {n_new} new and {n_replayed} replayed.")

            # The best team's median evaluation, so the metrics match the median ROC-AUC
            # it was ranked on.
            best_metrics: Dict[str, Any] = {}
            if successful:
                best_metrics = max(successful, key=lambda t: t.get_median_performance()).get_median_metrics()
            team_seconds = self.last_team_seconds or [0.0]
            summary = GenerationSummary(
                generation=generation,
                n_candidates=len(teams),
                n_new=n_new,
                n_carried=n_carried,
                n_replayed_run=self.last_replayed_run,
                n_replayed_external=self.last_replayed_external,
                n_successful=len(successful),
                n_failed=len(failed),
                n_survivors=len(self.population),
                successful_roc_auc=self._quartiles([t.get_median_performance() for t in successful]),
                successful_tokens=self._quartiles([t.get_median_token_count() for t in successful]),
                successful_cost_usd=self._quartiles([t.get_median_cost() for t in successful]),
                failed_tokens=self._quartiles([t.get_median_token_count() for t in failed]),
                failed_cost_usd=self._quartiles([t.get_median_cost() for t in failed]),
                total_tokens=total_tokens,
                total_cost_usd=total_cost,
                failed_total_tokens=failed_tokens,
                failed_total_cost_usd=failed_cost,
                failed_tokens_pct=100.0 * failed_tokens / total_tokens if total_tokens else 0.0,
                failed_cost_pct=100.0 * failed_cost / total_cost if total_cost else 0.0,
                generation_seconds=float(gen_seconds),
                evaluation_seconds=float(self.last_eval_seconds),
                team_seconds_mean=float(np.mean(team_seconds)),
                team_seconds_min=float(min(team_seconds)),
                team_seconds_max=float(max(team_seconds)),
                stage_times=dict(stage_times),
                best_metrics=dict(best_metrics),
                plot_path=plot_path)
            print_generation_summary(summary)
            append_generation_summary(summary, self.generation_csv)
        except Exception as exc:
            print(f"  [reporting] generation {generation} report failed: {exc!r}", flush=True)
        return

    def _generation_members(self, generation: int, candidates: List[Team]) -> List[GenerationMember]:
        """One row per candidate, in order: where it came from, its parents, its
        NSGA-II rank among the successful candidates, and whether it succeeded and
        survived (``self.population`` must already be the survivors).

        A candidate bred before ``generation`` is a carried-over survivor; the rest
        were bred this generation (or are initial teams, in generation -1) and give
        the lineage stamped on them by produce_offspring. That lineage is this
        breeding's, even when the genome's record was replayed from an earlier one.
        The ranks are over the same teams and scores survival selection ranked, so
        a survivor's front 0 is front zero of the population too.
        """
        survivor_ids = {self.archive.get_id(team) for team in self.population}
        viable = self.viable_positions(candidates)
        fronts: List[Optional[int]] = [None] * len(candidates)
        if viable:
            performances, objective2 = self.get_team_scores([candidates[i] for i in viable])
            for i, rank in zip(viable, nsga_tool.non_dominated_sorting(performances, objective2)):
                fronts[i] = int(rank)
        members = []
        for team, front in zip(candidates, fronts):
            bred_in = team.get_generation()
            assert bred_in <= generation, (
                f"A generation {generation} candidate was bred in generation {bred_in}.")
            if bred_in < generation:
                origin, parent_ids = "carried", None
            elif team.get_variation() is None:
                origin, parent_ids = "init", None
            else:
                origin, parent_ids = team.get_variation(), team.get_parent_ids()
            team_id = self.archive.get_id(team)
            members.append(GenerationMember(id=team_id, origin=origin, parent_ids=parent_ids,
                                            front=front, successful=front is not None,
                                            survived=team_id in survivor_ids))
        return members

    @staticmethod
    def _quartiles(values: List[float]) -> Optional[Quartiles]:
        """Min, Q1, median, Q3 and max of ``values`` (numpy's linear interpolation);
        None for an empty group."""
        if not values:
            return None
        q = np.percentile(np.asarray(values, dtype=np.float64), [0, 25, 50, 75, 100])
        return Quartiles(*(float(v) for v in q))

    def save_archive(self) -> None:
        """Write every genome evaluated so far (id, source, genome and full
        evaluation record) to ``archive.json`` in the run folder, replacing the
        previous save.

        The archive only grows, so the latest save holds the whole run so far; it
        can be passed back as an external archive to replay these evaluations.
        Errors are caught and printed here: a failed save must not take down a run
        that has already spent hours of LLM time.
        """
        try:
            self.archive.save_json(self.archive_path)
            print(f"  Saved archive ({len(self.archive)} teams) to {self.archive_path}.",
                  flush=True)
        except Exception as exc:
            print(f"  [reporting] archive save failed: {exc!r}", flush=True)
        return

    def _save_generation_frames(self, generation: int) -> Optional[str]:
        """Save ``generation``'s frame of each GIF from the run folder, and return its
        population frame (``None`` if saving failed). Errors are caught and printed
        here: reporting must never take down a run that has already spent hours of
        LLM time.
        """
        try:
            return save_generation_frames(self.run_dir, self.objective_2, self.gen_width,
                                          generation)[0]
        except Exception as exc:
            print(f"  [reporting] generation {generation} frames failed: {exc!r}", flush=True)
            return None

    def _save_phase_front(self, teams: List[Team], filename: str, title: str) -> None:
        """Plot every team of an evaluation phase, failures included, to
        ``Visuals/All_Front_PNG/{filename}``.

        The x-axis is the run's objective 2. Every team must already be archived
        (callers run this after ``_record_teams``) so its id can be drawn. Errors are
        caught and printed here: reporting must never take down a run that has
        already spent hours of LLM time.
        """
        try:
            _, front_data = self._front_plot_data(teams)
            plot_front(**front_data, objective_2=self.objective_2, title=title,
                       subtitle=f"{len(front_data['ids'])} teams (failures included)",
                       out_path=os.path.join(self.front_png_dir, filename))
        except Exception as exc:
            print(f"  [reporting] {filename} failed: {exc!r}", flush=True)
        return

    def _save_archive_front(self) -> None:
        """Plot every team in the archive, failures included, to
        ``Visuals/All_Front_PNG/all_front.png``.

        The end-of-run companion to the per-generation plots: one marker per
        unique genome the run evaluated, so the picture spans every generation.
        Errors are caught and printed by ``_save_phase_front``.
        """
        self._save_phase_front([record.team for record in self.archive.records()],
                               "all_front.png", title="Whole run: every evaluated team")
        return

    def _front_plot_data(self, teams: List[Team]) -> Tuple[List[Team], Dict[str, Any]]:
        """Everything ``plot_front`` needs about ``teams``, ready to draw, and the
        distinct teams those positions refer to.

        A genome present more than once (e.g. in both the population and the
        offspring) shares one archive id and is drawn once. The front is computed
        over the viable teams only (see ``viable_positions``) and ordered by
        objective 2; the failed teams are drawn but kept off it.
        """
        unique: Dict[int, Team] = {}
        for team in teams:
            unique.setdefault(self.archive.get_id(team), team)
        ids = list(unique)
        unique_teams = list(unique.values())
        performances, objective2 = self.get_team_scores(unique_teams)

        viable = np.asarray(self.viable_positions(unique_teams), dtype=np.int64)
        front = np.empty(0, dtype=np.int64)
        if viable.size:
            front = viable[nsga_tool.front_indices(performances[viable], objective2[viable])]
            front = front[np.argsort(objective2[front], kind="stable")]
        return unique_teams, {"objective2": objective2,
                              "performances": performances,
                              "ids": ids,
                              "front": front,
                              "dominated": np.setdiff1d(viable, front).astype(np.int64),
                              "failed": np.setdiff1d(np.arange(len(unique_teams)),
                                                     viable).astype(np.int64)}
