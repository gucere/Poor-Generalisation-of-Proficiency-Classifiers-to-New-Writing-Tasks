import argparse
import sys
import tempfile
import time
import zlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '0_Shared_Data_And_Splits'))
import ebm_workflow as wf  # noqa: E402

POST_TRAINING = wf.ROOT / '3_Language_Level_Prediction_with_LLMs/Code/Post-Training'
TOPICS, HOLDOUT = POST_TRAINING / 'Data/text_topics.csv', POST_TRAINING / 'topic_holdout.json'
FAMILIES = ('combined_without_length_correlated_features', 'combined_without_length_correlated_and_duplicate_features', 'combined')
LENGTH_FAMILIES = ('combined', 'combined_without_length_correlated_features',
                   'combined_without_length_correlated_and_duplicate_features', 'length_only')
LENGTH_BINS = (0, 50, 100, 150, 200, np.inf)
TASK_FOLDS, TASK_FAMILIES = 8, ('combined_without_length_correlated_features',)
TOPIC_FOLDER, LENGTH_FOLDER, TASK_FOLDER = HERE / '1_Topic_Check', HERE / '2_Length_Check', HERE / '3_Leave_One_Task_Out'
ROUNDS_FOLDER = HERE / '4_Training_Rounds'
ROUNDS_FAMILY, PAPER_ROUNDS, ROUND_GRID = 'combined_without_length_correlated_features', 200, (10, 25, 50, 100)
ROUNDS_METRICS = ('level_macro_f1', 'level_accuracy', 'cefr_macro_f1', 'cefr_accuracy', 'regression_mae')
SHUFFLES = ('shuffle_1', 'shuffle_2', 'shuffle_3')
# 'section_3' is the topic check's split. Its held-out half is the topic holdout of the LLM fine-tuning in section 3
# of the thesis project (LLM prediction), not a section of the paper.
SPLITS = ('section_3',) + SHUFFLES
DIRECTIONS = ('section_3_held_out', 'section_3_seen') + tuple(f'{split}_{half}' for split in SHUFFLES for half in ('a', 'b'))
SPLITS_FOLDER = HERE / '5_Training_Rounds_New_Topic_Splits'
SPLIT_FOLDERS = {split: SPLITS_FOLDER / f'{number}_Topic_Split_{number}' for number, split in enumerate(SHUFFLES, start=1)}
ANALYSIS_FOLDER = SPLITS_FOLDER / '4_Analysis'


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def write_csv(path, table):
    temporary = Path(f'{path}.tmp')
    table.to_csv(temporary, index=False)
    wf.os.replace(temporary, path)


def subset_metrics(predictions, manifest, algorithm):
    """The pipeline's metrics for a subset of the saved predictions of a level model."""
    labels = manifest['labels']
    metrics = {'texts': len(predictions)}
    for target, truth, predicted, names in (('level', 'level', 'predicted_level', list(range(1, 16))),
                                            ('cefr', 'cefr_level', 'predicted_cefr', labels)):
        values = wf.class_metrics(predictions[truth], predictions[predicted], names)
        metrics.update({f'{target}_{name}': values[name] for name in wf.CLASS_METRICS})
    if algorithm == 'regressor':
        errors = predictions.predicted_level_continuous - predictions.level
        metrics.update({'regression_mae': float(errors.abs().mean()), 'regression_rmse': float(np.sqrt((errors ** 2).mean()))})
    return metrics


def topic_groups(folder=None):
    """Topic of every text, and the two topic groups: section 3's held-out topics and the others. `folder` reads the
    two files from another folder (--topic-folder), e.g. a copy of the LLM training data."""
    topics_path, holdout_path = (folder / TOPICS.name, folder / HOLDOUT.name) if folder else (TOPICS, HOLDOUT)
    holdout = wf.read_json(holdout_path)
    if wf.digest(topics_path) != holdout['topic_map_sha256']:
        raise ValueError(f'{topics_path.name} differs from the topic map used for {holdout_path.name}.')
    topics = pd.read_csv(topics_path, dtype={'topic_id': str}, encoding='utf-8-sig')[['text_id', 'topic_id']]
    held_out = set(holdout['held_out_topics'])
    return topics, {'section_3_held_out': held_out, 'section_3_seen': set(topics.topic_id) - held_out}


def topic_check(args):
    manifest, frame = wf.verify_shared(args.shared)
    topics, groups = topic_groups(args.topic_folder)
    frame = frame.merge(topics, on='text_id', how='left', validate='one_to_one')
    if frame.topic_id.isna().any():
        raise ValueError(f'{int(frame.topic_id.isna().sum())} texts have no topic.')
    definitions = wf.experiment_definitions(argparse.Namespace(shared=args.shared, task='level', use_ranked=False, ranking=None), manifest)
    validation = frame[frame.split.eq('validation')].sort_values('text_id')
    runs = TOPIC_FOLDER / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in args.families or FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            for trained_on, topic_set in groups.items():
                name = f'{algorithm}__{family}__trained_on_{trained_on}'
                training = frame[frame.split.eq('train') & frame.topic_id.isin(topic_set)].sort_values('text_id')
                seen = validation.topic_id.isin(topic_set)
                log(f'{name}: {len(training):,} training texts; validation {int(seen.sum()):,} seen / {int((~seen).sum()):,} unseen topics')
                if args.dry_run:
                    continue
                config = {'dataset_id': wf.token(manifest), 'family': family, 'algorithm': algorithm, 'trained_on': trained_on,
                          'topics': sorted(topic_set, key=lambda topic: int(topic)), 'columns_sha256': wf.token(definitions[family]['columns']),
                          'rounds': args.rounds, 'min_observations': args.min_observations, 'seed': manifest['seed']}
                folder = runs / name
                marker = folder / 'completed.json'
                if not (marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)):
                    with wf.training_lock(folder), tempfile.TemporaryDirectory(prefix='.fit_', dir=runs) as temporary:
                        if not (marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)):
                            predictions = wf.fit_run(manifest, training, validation, definitions[family]['columns'], 'level', algorithm, family,
                                                     args.rounds, args.min_observations, args.workers, args.memory, Path(temporary), name)[4]
                            predictions = predictions.merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one')
                            folder.mkdir(exist_ok=True)
                            predictions.to_csv(folder / 'validation_predictions.csv', index=False)
                            wf.write_json(folder / 'config.json', config)
                            wf.write_json(marker, {'configuration_id': wf.token(config), 'training_texts': len(training)})
                predictions = pd.read_csv(folder / 'validation_predictions.csv', dtype={'topic_id': str, 'cefr_level': str,
                                                                                         'predicted_cefr': str})
                main = pd.read_csv(wf.BASE / '3_Level_Prediction/1_Intermediate_Calculations' / f'{algorithm}__{family}' /
                                   'validation_predictions.csv', dtype={'cefr_level': str, 'predicted_cefr': str}) \
                    .merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one')
                for model, table in (('topic_restricted', predictions), ('main_model_all_topics', main)):
                    for subset, mask in (('seen_topics', table.topic_id.isin(topic_set)), ('unseen_topics', ~table.topic_id.isin(topic_set))):
                        rows.append({'family': family, 'algorithm': algorithm, 'trained_on': trained_on, 'model': model,
                                     'topics': subset, **subset_metrics(table[mask], manifest, algorithm)})
    if args.dry_run or args.families or not rows:
        return  # the tables are written by a run over all models
    results = pd.DataFrame(rows)
    write_csv(TOPIC_FOLDER / 'topic_check_results.csv', results)
    metrics = [name for name in ('level_macro_f1', 'level_accuracy', 'level_quadratic_weighted_kappa', 'cefr_macro_f1',
                                 'regression_mae') if name in results]
    wide = results.pivot_table(index=['family', 'algorithm', 'trained_on', 'model'], columns='topics', values=metrics)
    gaps = pd.DataFrame({metric: wide[(metric, 'seen_topics')] - wide[(metric, 'unseen_topics')] for metric in metrics}).reset_index()
    # Averaged over both directions, so topics that are simply easier or harder cancel out.
    summary = gaps.groupby(['family', 'algorithm', 'model'], as_index=False)[metrics].mean()
    restricted = summary[summary.model.eq('topic_restricted')].set_index(['family', 'algorithm'])[metrics]
    reference = summary[summary.model.eq('main_model_all_topics')].set_index(['family', 'algorithm'])[metrics]
    effect = (restricted - reference).reset_index().assign(model='topic_effect: restricted gap minus main-model gap')
    write_csv(TOPIC_FOLDER / 'topic_check_gaps.csv', pd.concat([summary, effect], ignore_index=True).rename(
        columns={metric: f'{metric}_seen_minus_unseen' for metric in metrics}))
    log(f'Saved {TOPIC_FOLDER}')


def unseen_halves(topic_levels, seen_topics):
    """Split each level's unseen tasks in two halves by topic number, so the number of rounds can be chosen on one
    half and reported on the other."""
    by_level = {}
    for topic, level in topic_levels.items():
        if topic not in seen_topics:
            by_level.setdefault(level, []).append(topic)
    first, second = set(), set()
    for topics in by_level.values():
        if len(topics) < 2:
            raise ValueError('Every level needs at least two unseen tasks.')
        topics = sorted(topics, key=int)
        first.update(topics[:len(topics) // 2])
        second.update(topics[len(topics) // 2:])
    return first, second


def topic_splits(groups, topic_levels, seed):
    """The training topic splits, each as its two halves: section 3's split and its reverse, and the shuffles, new
    random 4/4 splits of each level's topics drawn with fixed seeds, each with its reverse."""
    by_level = {}
    for topic, level in sorted(topic_levels.items(), key=lambda item: int(item[0])):
        by_level.setdefault(level, []).append(topic)
    splits = {'section_3': {'section_3_held_out': groups['section_3_held_out'], 'section_3_seen': groups['section_3_seen']}}
    for number, split in enumerate(SHUFFLES, start=1):
        generator = np.random.default_rng([seed, number])
        half = set()
        for level in sorted(by_level):
            half.update(str(topic) for topic in generator.choice(by_level[level], len(by_level[level]) // 2, replace=False))
        splits[split] = {f'{split}_a': half, f'{split}_b': set(topic_levels) - half}
    halves = [frozenset(half) for halves_of in splits.values() for half in halves_of.values()]
    if len(set(halves)) != len(halves):
        raise ValueError('Two training topic splits are the same.')
    return splits


def rounds_run(manifest, frame, validation, definitions, algorithm, trained_on, topic_set, rounds, args, train=True):
    """Validation predictions of a topic-restricted main model trained with the given number of boosting rounds.
    Section 3's 200-round models are the topic check's own runs, reused and never retrained here. Without `train`, a
    model that is not trained yet is an error instead of being trained."""
    name = f'{algorithm}__{ROUNDS_FAMILY}__trained_on_{trained_on}'
    training = frame[frame.split.eq('train') & frame.topic_id.isin(topic_set)].sort_values('text_id')
    config = {'dataset_id': wf.token(manifest), 'family': ROUNDS_FAMILY, 'algorithm': algorithm, 'trained_on': trained_on,
              'topics': sorted(topic_set, key=lambda topic: int(topic)), 'columns_sha256': wf.token(definitions[ROUNDS_FAMILY]['columns']),
              'rounds': rounds, 'min_observations': args.min_observations, 'seed': manifest['seed']}
    folder, reused = run_folder(name, trained_on, rounds)
    marker = folder / 'completed.json'
    done = lambda: marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)
    if reused and not done():
        raise ValueError(f'{name}: the topic check\'s 200-round run is missing or was made with other settings; run the topic check first.')
    if args.dry_run:
        log(f'{name}, {rounds} rounds: {len(training):,} training texts' + (' (finished)' if done() else ''))
        return None
    if not train and not done():
        raise ValueError(f'{name}, {rounds} rounds: not trained yet; run the rounds command first.')
    if not done():
        folder.parent.mkdir(parents=True, exist_ok=True)
        with wf.training_lock(folder), tempfile.TemporaryDirectory(prefix='.fit_', dir=folder.parent) as temporary:
            if not done():
                started = time.time()
                log(f'{name}: training with {rounds} rounds')
                predictions = wf.fit_run(manifest, training, validation, definitions[ROUNDS_FAMILY]['columns'], 'level', algorithm,
                                         ROUNDS_FAMILY, rounds, args.min_observations, args.workers, args.memory, Path(temporary),
                                         f'{name}__rounds_{rounds}')[4]
                predictions = predictions.merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one')
                folder.mkdir(exist_ok=True)
                predictions.to_csv(folder / 'validation_predictions.csv', index=False)
                wf.write_json(folder / 'config.json', config)
                wf.write_json(marker, {'configuration_id': wf.token(config), 'training_texts': len(training),
                                       'minutes': round((time.time() - started) / 60, 1)})
                log(f'{name}, {rounds} rounds: done in {(time.time() - started) / 60:.1f} min')
    return pd.read_csv(folder / 'validation_predictions.csv', dtype={'topic_id': str, 'cefr_level': str, 'predicted_cefr': str})


def rounds_figure(curve, grid, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    panels = (('level_macro_f1', 'level macro F1'), ('cefr_accuracy', 'CEFR accuracy'))
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    for row, algorithm in enumerate(('classifier', 'regressor')):
        for column, (metric, label) in enumerate(panels):
            axis = axes[row, column]
            for tasks, style in (('seen_tasks', '-o'), ('unseen_tasks', '--s')):
                line = curve[curve.algorithm.eq(algorithm) & curve.tasks.eq(tasks)].sort_values('rounds')
                axis.plot(line.rounds, line[metric], style, markersize=4, label=tasks.replace('_', ' '))
            axis.set_xscale('log')
            axis.set_xticks(grid)
            axis.set_xticklabels([str(rounds) for rounds in grid])
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_title(f'Main {algorithm}: {label}', fontsize=9)
            axis.grid(alpha=.3)
    for axis in axes[1]:
        axis.set_xlabel('Boosting rounds (the paper uses 200)')
    axes[0, 0].legend(fontsize=8, frameon=False)
    figure.suptitle('Models trained on 4 of the 8 writing tasks per level (mean of both halves)', fontsize=9)
    figure.tight_layout()
    figure.savefig(path.with_suffix('.png'), dpi=200)
    figure.savefig(path.with_suffix('.pdf'), metadata={'CreationDate': None})
    plt.close(figure)


def rounds_models(args, train=True):
    """The validation predictions of the training-rounds models of the splits in --splits, keyed by (algorithm,
    trained_on, rounds), with those splits, each topic's level and the round counts. Missing models are trained, or are
    an error without `train`."""
    manifest, frame = wf.verify_shared(args.shared)
    topics, groups = topic_groups(args.topic_folder)
    frame = frame.merge(topics, on='text_id', how='left', validate='one_to_one')
    if frame.topic_id.isna().any():
        raise ValueError(f'{int(frame.topic_id.isna().sum())} texts have no topic.')
    levels = frame.groupby('topic_id')['level'].unique()
    if levels.map(len).ne(1).any():
        raise ValueError('A writing task belongs to more than one level.')
    topic_levels = levels.map(lambda values: int(values[0])).to_dict()
    definitions = wf.experiment_definitions(argparse.Namespace(shared=args.shared, task='level', use_ranked=False, ranking=None), manifest)
    validation = frame[frame.split.eq('validation')].sort_values('text_id')
    grid = sorted(set(args.round_grid) | {PAPER_ROUNDS})
    every_split = topic_splits(groups, topic_levels, manifest['seed'])
    splits = {split: every_split[split] for split in args.splits}
    if not args.dry_run:
        show_names = not getattr(args, 'no_topic_names', False)
        write_split_records(every_split, topic_levels, topic_names(args.topic_folder) if show_names else None, manifest['seed'], show_names)
    directions = [(name, topic_set) for halves in splits.values() for name, topic_set in halves.items()
                  if not args.trained_on or name in args.trained_on]
    if args.trained_on and set(args.trained_on) - {name for name, _ in directions}:
        raise ValueError(f'--trained-on {sorted(set(args.trained_on) - {name for name, _ in directions})} is not part of --splits {args.splits}.')
    if args.dry_run:
        for name, topic_set in directions:
            log(f'{name}: trains on topics {" ".join(sorted(topic_set, key=int))}')
    predictions = {}
    for algorithm in ('classifier', 'regressor'):
        for trained_on, topic_set in directions:
            for rounds in grid:
                table = rounds_run(manifest, frame, validation, definitions, algorithm, trained_on, topic_set, rounds, args, train)
                if table is not None:
                    predictions[algorithm, trained_on, rounds] = table
    return manifest, splits, topic_levels, grid, predictions


def rounds_check(args):
    """The topic check's main models with fewer boosting rounds: does stopping earlier, chosen on writing tasks not
    seen in training, generalise better to new tasks? Trains the models of the splits in --splits; the tables below
    are section 3's split only."""
    manifest, splits, topic_levels, grid, predictions = rounds_models(args)
    groups = splits.get('section_3')
    if (args.dry_run or args.trained_on or groups is None
            or any((algorithm, half, rounds) not in predictions for algorithm in ('classifier', 'regressor') for half in groups for rounds in grid)):
        return  # section 3's tables are written by a run over both of its halves
    rows, choices = [], []
    for algorithm in ('classifier', 'regressor'):
        for trained_on, topic_set in groups.items():
            for rounds in grid:
                table = predictions[algorithm, trained_on, rounds]
                for tasks, mask in (('seen_tasks', table.topic_id.isin(topic_set)), ('unseen_tasks', ~table.topic_id.isin(topic_set))):
                    rows.append({'algorithm': algorithm, 'trained_on': trained_on, 'rounds': rounds, 'tasks': tasks,
                                 **subset_metrics(table[mask], manifest, algorithm)})
            first, second = unseen_halves(topic_levels, topic_set)
            for chosen_on, chosen, reported in (('first_half', first, second), ('second_half', second, first)):
                score = {rounds: subset_metrics(predictions[algorithm, trained_on, rounds].pipe(lambda t: t[t.topic_id.isin(chosen)]),
                                                manifest, algorithm)[args.select_by] for rounds in grid}
                lower_is_better = args.select_by.endswith('mae')
                # Ties go to more rounds, the paper's default direction.
                selected = (min if lower_is_better else max)(sorted(grid, reverse=True), key=lambda rounds: score[rounds])
                row = {'algorithm': algorithm, 'trained_on': trained_on, 'rounds_chosen_on': f'unseen_tasks_{chosen_on}',
                       'selected_by': args.select_by, 'selected_rounds': selected}
                for label, rounds in (('selected', selected), ('200_rounds', PAPER_ROUNDS)):
                    table = predictions[algorithm, trained_on, rounds]
                    for tasks, subset in (('unseen_reported_half', table[table.topic_id.isin(reported)]),
                                          ('seen', table[table.topic_id.isin(topic_set)])):
                        values = subset_metrics(subset, manifest, algorithm)
                        row.update({f'{metric}_{tasks}_{label}': values[metric] for metric in ROUNDS_METRICS if metric in values})
                choices.append(row)
    results, selection = pd.DataFrame(rows), pd.DataFrame(choices)
    metrics = [metric for metric in ROUNDS_METRICS if metric in results]
    curve = results.groupby(['algorithm', 'rounds', 'tasks'], as_index=False)[metrics].mean()
    numbers = [column for column in selection.columns if column.endswith(('_selected', '_200_rounds'))]
    summary = selection.groupby('algorithm', as_index=False)[numbers].mean()
    summary.insert(1, 'selected_rounds', selection.groupby('algorithm').selected_rounds.agg(lambda values: '/'.join(map(str, values))).values)
    ROUNDS_FOLDER.mkdir(parents=True, exist_ok=True)
    write_csv(ROUNDS_FOLDER / 'training_rounds_results.csv', results)
    write_csv(ROUNDS_FOLDER / 'training_rounds_curve.csv', curve)
    write_csv(ROUNDS_FOLDER / 'training_rounds_selection.csv', selection)
    write_csv(ROUNDS_FOLDER / 'training_rounds_selection_summary.csv', summary)
    rounds_figure(curve, grid, ROUNDS_FOLDER / 'training_rounds_curve')
    log(f'Saved {ROUNDS_FOLDER}')


def choosing_tasks(topic_levels, seen_topics, generator):
    """Half of each level's unseen tasks, drawn at random, for choosing the number of rounds; the other half is
    for scoring."""
    by_level = {}
    for topic, level in sorted(topic_levels.items(), key=lambda item: int(item[0])):
        if topic not in seen_topics:
            by_level.setdefault(level, []).append(topic)
    chosen = set()
    for level in sorted(by_level):
        if len(by_level[level]) < 2:
            raise ValueError('Every level needs at least two unseen tasks.')
        chosen.update(str(topic) for topic in generator.choice(by_level[level], len(by_level[level]) // 2, replace=False))
    return chosen


def label_codes(table, cefr_labels, algorithm):
    """Each text's true and predicted level and CEFR band as label positions, and the regressor's absolute error,
    for quick_metrics."""
    codes = {}
    for target, truth, predicted, labels in (('level', 'level', 'predicted_level', list(range(1, 16))),
                                             ('cefr', 'cefr_level', 'predicted_cefr', list(cefr_labels))):
        positions = {label: index for index, label in enumerate(labels)}
        actual, guessed = table[truth].map(positions), table[predicted].map(positions)
        if actual.isna().any() or guessed.isna().any():
            raise ValueError(f'Unknown {target} label.')
        codes[target] = (actual.to_numpy(int), guessed.to_numpy(int), len(labels))
    if algorithm == 'regressor':
        codes['absolute_error'] = (table.predicted_level_continuous - table.level).abs().to_numpy()
    return codes


def quick_metrics(codes, mask):
    """Level and CEFR macro F1 and accuracy, and the regressor's MAE, of the texts in `mask`: the values of
    subset_metrics without its full reports, so that many repeated splits stay fast (the tests check that they match)."""
    if not mask.any():
        raise ValueError('No texts to score.')
    metrics = {}
    for target in ('level', 'cefr'):
        actual, guessed, size = codes[target]
        confusion = np.bincount(actual[mask] * size + guessed[mask], minlength=size * size).reshape(size, size)
        hits = np.diag(confusion)
        counted = confusion.sum(axis=0) + confusion.sum(axis=1)  # 2 TP + FP + FN of each label
        f1 = np.divide(2 * hits, counted, out=np.zeros(size), where=counted > 0)
        metrics.update({f'{target}_macro_f1': float(f1.mean()), f'{target}_accuracy': float(hits.sum() / mask.sum())})
    if 'absolute_error' in codes:
        metrics['regression_mae'] = float(codes['absolute_error'][mask].mean())
    return metrics


def selection_check(args):
    """Exploratory follow-up of the training-rounds check (added after the paper, on data already examined): does
    choosing the number of rounds on validation texts of unseen writing tasks pick a model that transfers better than
    choosing it on validation texts of the tasks the model was trained on? Both choices pick from the same models and
    use texts of the same validation learners. Both are scored on the other validation learners' texts: of unseen
    tasks not used for choosing, and of the seen tasks. Each repeat draws a new half of the validation learners and a
    new half of each level's unseen tasks for choosing. With several --splits, each training topic split is summarised
    on its own and all together. Only reads the saved predictions of the rounds command. This command covers section
    3's split; the splits command repeats it over all four splits."""
    manifest, splits, topic_levels, grid, predictions = rounds_models(argparse.Namespace(**{**vars(args), 'splits': ['section_3']}), train=False)
    if args.dry_run:
        return
    if len(predictions) < 2 * sum(len(halves) for halves in splits.values()) * len(grid):
        raise ValueError('Run the rounds command over both halves of every split first.')
    tables = selection_tables(predictions, splits, topic_levels, grid, manifest['labels'], manifest['seed'], args.repeats, args.select_by)
    ROUNDS_FOLDER.mkdir(parents=True, exist_ok=True)
    for name, table in zip(('repeats', 'summary', 'rounds', 'curve'), tables):
        write_csv(ROUNDS_FOLDER / f'selection_by_topic_{name}.csv', table)
    log(f'Saved the selection comparison ({args.repeats} repeats) in {ROUNDS_FOLDER}')


def selection_tables(predictions, splits, topic_levels, grid, labels, seed, repeats, select_by):
    """The selection comparison from the saved predictions: one row per repeat, model and training half; the summary
    over the repeats, per split and, with several splits, over all of them; how often each number of rounds is chosen;
    and every model's scores on seen and unseen tasks by number of rounds. All splits share each repeat's learner
    halves. Section 3's choosing tasks come from the same random stream as the learner halves and every other split's
    from a stream of its own, so adding splits leaves section 3's results unchanged."""
    halves = {half: (split, topic_set) for split, of_split in splits.items() for half, topic_set in of_split.items()}
    reference = predictions['classifier', next(iter(halves)), PAPER_ROUNDS].sort_values('text_id', ignore_index=True)
    codes = {}
    for key, table in predictions.items():
        table = table.sort_values('text_id', ignore_index=True)
        if not table.text_id.equals(reference.text_id):
            raise ValueError(f'{key}: the saved predictions cover other validation texts.')
        codes[key] = label_codes(table, labels, key[0])
    learners = np.sort(reference.learner_id.unique())
    seen = {half: reference.topic_id.isin(topic_set).to_numpy() for half, (_, topic_set) in halves.items()}
    lower_is_better = select_by.endswith('mae')
    generator = np.random.default_rng(seed)
    streams = {split: generator if split == 'section_3' else np.random.default_rng([seed, zlib.crc32(split.encode())]) for split in splits}
    rows = []
    for repeat in range(1, repeats + 1):
        choosing = reference.learner_id.isin(set(generator.permutation(learners)[:len(learners) // 2].tolist())).to_numpy()
        subsets, picked = {}, {}
        for half, (split, topic_set) in halves.items():
            seen_by_model = seen[half]
            picked[half] = choosing_tasks(topic_levels, topic_set, streams[split])
            unseen_choosing = reference.topic_id.isin(picked[half]).to_numpy()
            subsets[half] = {'choose_seen': choosing & seen_by_model, 'choose_unseen': choosing & unseen_choosing,
                             'score_unseen_tasks': ~choosing & ~seen_by_model & ~unseen_choosing, 'score_seen_tasks': ~choosing & seen_by_model}
        for algorithm in ('classifier', 'regressor'):
            for half, (split, _) in halves.items():
                scores = {rounds: {name: quick_metrics(codes[algorithm, half, rounds], mask) for name, mask in subsets[half].items()}
                          for rounds in grid}
                # Ties go to more rounds, the paper's default direction.
                pick = lambda subset: (min if lower_is_better else max)(sorted(grid, reverse=True),
                                                                        key=lambda rounds: scores[rounds][subset][select_by])
                choices = {'seen_choice': pick('choose_seen'), 'unseen_choice': pick('choose_unseen'), '200_rounds': PAPER_ROUNDS}
                row = {'split': split, 'repeat': repeat, 'algorithm': algorithm, 'trained_on': half, 'selected_by': select_by,
                       'seen_choice_rounds': choices['seen_choice'], 'unseen_choice_rounds': choices['unseen_choice'],
                       'choosing_topics': ' '.join(sorted(picked[half], key=int)),
                       **{f'texts_{name}': int(mask.sum()) for name, mask in subsets[half].items()}}
                for approach, rounds in choices.items():
                    for scored_on in ('unseen_tasks', 'seen_tasks'):
                        row.update({f'{metric}_{scored_on}_{approach}': value for metric, value in scores[rounds][f'score_{scored_on}'].items()})
                rows.append(row)
    results = pd.DataFrame(rows)
    approaches = ('seen_choice', 'unseen_choice', '200_rounds')
    values = [column for column in results.columns if column.endswith(approaches)]

    def summarise(table, label):
        # Each repeat: the mean over the training halves, as in the other tables of this check.
        per_repeat = table.groupby(['algorithm', 'repeat'])[values].mean()
        summary = []
        for algorithm, of_algorithm in per_repeat.groupby(level='algorithm'):
            rows_of = table[table.algorithm.eq(algorithm)]
            same_rounds = float(rows_of.seen_choice_rounds.eq(rows_of.unseen_choice_rounds).mean())
            for scored_on in ('unseen_tasks', 'seen_tasks'):
                for metric in ROUNDS_METRICS:
                    if f'{metric}_{scored_on}_200_rounds' not in of_algorithm or of_algorithm[f'{metric}_{scored_on}_200_rounds'].isna().all():
                        continue
                    seen_choice, unseen_choice, fixed = (of_algorithm[f'{metric}_{scored_on}_{approach}'] for approach in approaches)
                    difference = unseen_choice - seen_choice
                    better = difference.lt(0) if metric.endswith('mae') else difference.gt(0)
                    summary.append({'split': label, 'algorithm': algorithm, 'scored_on': scored_on, 'metric': metric,
                                    'seen_choice': seen_choice.mean(), 'unseen_choice': unseen_choice.mean(), '200_rounds': fixed.mean(),
                                    'unseen_minus_seen_choice': difference.mean(), 'difference_2.5%': difference.quantile(.025),
                                    'difference_97.5%': difference.quantile(.975), 'share_unseen_choice_better': float(better.mean()),
                                    'share_equal': float(difference.eq(0).mean()), 'share_same_rounds': same_rounds, 'repeats': len(difference)})
        return summary

    def chosen_rounds(table, label):
        chosen = table.melt(id_vars=['algorithm'], value_vars=['seen_choice_rounds', 'unseen_choice_rounds'], var_name='choice', value_name='rounds')
        chosen['choice'] = chosen.choice.str.removesuffix('_rounds')
        counts = chosen.groupby(['algorithm', 'choice', 'rounds']).size().rename('choices').reset_index()
        counts['share'] = counts.choices / counts.groupby(['algorithm', 'choice']).choices.transform('sum')
        return counts.assign(split=label)

    labelled = [(split, results[results.split.eq(split)]) for split in splits] + ([('all', results)] if len(splits) > 1 else [])
    summary = pd.DataFrame([row for label, table in labelled for row in summarise(table, label)])
    counts = pd.concat([chosen_rounds(table, label) for label, table in labelled], ignore_index=True)
    counts = counts[['split'] + [column for column in counts.columns if column != 'split']]
    every = pd.DataFrame([{'split': split, 'trained_on': half, 'algorithm': algorithm, 'rounds': rounds, 'tasks': tasks,
                           **quick_metrics(codes[algorithm, half, rounds], seen[half] if tasks == 'seen_tasks' else ~seen[half])}
                          for half, (split, _) in halves.items() for algorithm in ('classifier', 'regressor') for rounds in grid
                          for tasks in ('seen_tasks', 'unseen_tasks')])
    metrics = [metric for metric in ROUNDS_METRICS if metric in every]
    # The mean over each split's two halves and, with several splits, over all halves.
    curve = [every.groupby(['split', 'algorithm', 'rounds', 'tasks'], as_index=False)[metrics].mean()]
    if len(splits) > 1:
        curve.append(every.groupby(['algorithm', 'rounds', 'tasks'], as_index=False)[metrics].mean().assign(split='all'))
    curve = pd.concat(curve, ignore_index=True)[['split', 'algorithm', 'rounds', 'tasks'] + metrics]
    return results, summary, counts, curve


def topic_names(folder=None):
    """Each topic's name, from the topic map."""
    path = folder / TOPICS.name if folder else TOPICS
    table = pd.read_csv(path, dtype={'topic_id': str}, usecols=['topic_id', 'topic'], encoding='utf-8-sig').drop_duplicates()
    if table.topic_id.duplicated().any():
        raise ValueError('A topic has more than one name.')
    return dict(zip(table.topic_id, table.topic))


def topic_label(topic, names, show_names):
    return f'{topic} ({names[topic]})' if show_names else topic


def split_record(split, halves, topic_levels, names, seed, show_names=True):
    """Which topics each half of a shuffled split trains on, per level, as written to its topic_split.json."""
    levels = sorted(set(topic_levels.values()))
    return {'split': split, 'seed': [seed, SHUFFLES.index(split) + 1],
            'halves': {half: {str(level): [{'topic_id': topic, **({'topic': names[topic]} if show_names else {})}
                                           for topic in sorted((topic for topic in topics if topic_levels[topic] == level), key=int)]
                              for level in levels} for half, topics in halves.items()}}


def write_split_records(every_split, topic_levels, names, seed, show_names=True):
    """Write each shuffled split's topic_split.json, unless it already holds the same split."""
    for split in SHUFFLES:
        path = SPLIT_FOLDERS[split] / 'topic_split.json'
        record = split_record(split, every_split[split], topic_levels, names, seed, show_names)
        if not path.is_file() or wf.read_json(path) != record:
            wf.write_json(path, record)


def run_folder(name, trained_on, rounds):
    """Where a training-rounds model is saved, and whether it is one of the topic check's runs (section 3's
    200-round models), which are reused."""
    reused = rounds == PAPER_ROUNDS and trained_on.startswith('section_3')
    split = trained_on.rsplit('_', 1)[0]
    if reused:
        return TOPIC_FOLDER / 'runs' / name, True
    return (SPLIT_FOLDERS[split] if split in SPLIT_FOLDERS else ROUNDS_FOLDER) / 'runs' / f'{name}__rounds_{rounds}', False


def topic_assignments(splits, topic_levels, names, show_names=True):
    """One row per topic: its level and, for every split, the half whose models train on it."""
    rows = []
    for topic in sorted(topic_levels, key=lambda topic: (topic_levels[topic], int(topic))):
        row = {'level': topic_levels[topic], 'topic_id': topic, **({'topic': names[topic]} if show_names else {})}
        for split, halves in splits.items():
            row[split] = next(half.removeprefix(f'{split}_') for half, topics in halves.items() if topic in topics)
        rows.append(row)
    return pd.DataFrame(rows)


def splits_check(args):
    """Exploratory, added after the paper: the selection comparison repeated over the topic check's split and three
    new random topic splits (shuffle_1 to shuffle_3). Writes the tables and figures of 4_Analysis and every split
    folder's README. Only reads the saved predictions of the rounds command."""
    manifest, splits, topic_levels, grid, predictions = rounds_models(argparse.Namespace(**{**vars(args), 'splits': list(SPLITS)}), train=False)
    if args.dry_run:
        return
    if len(predictions) < 2 * sum(len(halves) for halves in splits.values()) * len(grid):
        raise ValueError(f'Train every split first: rounds --splits {" ".join(SHUFFLES)}')
    results, summary, counts, curve = selection_tables(predictions, splits, topic_levels, grid, manifest['labels'], manifest['seed'],
                                                       args.repeats, args.select_by)
    show_names = not args.no_topic_names
    names = topic_names(args.topic_folder) if show_names else None
    ANALYSIS_FOLDER.mkdir(parents=True, exist_ok=True)
    for name, table in (('summary', summary), ('chosen_rounds', counts), ('curve', curve), ('repeats', results),
                        ('topic_assignments', topic_assignments(splits, topic_levels, names, show_names))):
        write_csv(ANALYSIS_FOLDER / f'{name}.csv', table)
    splits_figure(curve, grid, ANALYSIS_FOLDER / 'curve')
    choice_figure(summary, ANALYSIS_FOLDER / 'choice_difference')
    for split in SHUFFLES:
        text = split_readme(split, splits[split], topic_levels, names, summary, manifest, show_names)
        (SPLIT_FOLDERS[split] / 'README.md').write_text(text, encoding='utf-8', newline='\n')
    log(f'Saved {ANALYSIS_FOLDER} and the README of every split folder')


SPLIT_LABELS = {'section_3': "Topic check's split", 'shuffle_1': 'Topic split 1', 'shuffle_2': 'Topic split 2',
                'shuffle_3': 'Topic split 3', 'all': 'All four splits'}


def splits_figure(curve, grid, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    panels = (('level_macro_f1', 'level macro F1'), ('cefr_accuracy', 'CEFR accuracy'))
    splits = [split for split in curve.split.unique() if split != 'all']
    colours = dict(zip(splits, plt.rcParams['axes.prop_cycle'].by_key()['color']))
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for row, algorithm in enumerate(('classifier', 'regressor')):
        for column, (metric, label) in enumerate(panels):
            axis = axes[row, column]
            for split in splits:
                for tasks, style in (('seen_tasks', '-'), ('unseen_tasks', '--')):
                    line = curve[curve.split.eq(split) & curve.algorithm.eq(algorithm) & curve.tasks.eq(tasks)].sort_values('rounds')
                    axis.plot(line.rounds, line[metric], style, marker='o', markersize=3, color=colours[split])
            axis.set_xscale('log')
            axis.set_xticks(grid)
            axis.set_xticklabels([str(rounds) for rounds in grid])
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_title(f'Main {algorithm}: {label}', fontsize=9)
            axis.grid(alpha=.3)
    for axis in axes[1]:
        axis.set_xlabel('Boosting rounds (the paper uses 200)')
    handles = [Line2D([], [], color=colours[split], label=SPLIT_LABELS[split]) for split in splits]
    handles += [Line2D([], [], color='grey', linestyle='-', label='seen topics'), Line2D([], [], color='grey', linestyle='--', label='unseen topics')]
    figure.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, frameon=False)
    figure.suptitle('Models trained on 4 of the 8 writing topics per level (mean of the two halves of each split)', fontsize=9)
    figure.tight_layout(rect=(0, .1, 1, 1))
    figure.savefig(path.with_suffix('.png'), dpi=200)
    figure.savefig(path.with_suffix('.pdf'), metadata={'CreationDate': None})
    plt.close(figure)


def choice_figure(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    rows = summary[summary.metric.eq('level_macro_f1') & summary.scored_on.eq('unseen_tasks')]
    order = [split for split in SPLIT_LABELS if split in set(rows.split)]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for axis, algorithm in zip(axes, ('classifier', 'regressor')):
        part = rows[rows.algorithm.eq(algorithm)].set_index('split').loc[order]
        y = np.arange(len(order))[::-1]
        middle = part.unseen_minus_seen_choice
        axis.errorbar(middle, y, xerr=[middle - part['difference_2.5%'], part['difference_97.5%'] - middle], fmt='o', color='black', capsize=3)
        axis.axvline(0, color='grey', linewidth=.8)
        axis.set_yticks(y)
        axis.set_yticklabels([SPLIT_LABELS[split] for split in order], fontsize=8)
        axis.set_title(f'Main {algorithm}', fontsize=9)
        axis.set_xlabel('Level macro F1 on unseen topics:\nrounds chosen on unseen minus on seen topics', fontsize=8)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.tick_params(axis='x', labelsize=8)
        axis.grid(axis='x', alpha=.3)
    figure.suptitle(f'Mean over {int(rows.repeats.iloc[0])} repeats, with the 2.5–97.5% range of the repeats', fontsize=9)
    figure.tight_layout()
    figure.savefig(path.with_suffix('.png'), dpi=200)
    figure.savefig(path.with_suffix('.pdf'), metadata={'CreationDate': None})
    plt.close(figure)


def split_readme(split, halves, topic_levels, names, summary, manifest, show_names=True):
    """The README of a shuffled split's folder: which topics each half trains on, its models and its results."""
    number = SHUFFLES.index(split) + 1
    (half_a, topics_a), (half_b, topics_b) = halves.items()
    lines = [f'# Topic split {number}', '',
             f'Part of [5_Training_Rounds_New_Topic_Splits](../README.md), an exploratory check added after the paper. '
             f'Topic split {number} (`{split}` in the code and the file names) is a random split of each course level\'s 8 writing '
             f'topics into two halves of 4, drawn with the seed [{manifest["seed"]}, {number}]. The models of half a (`{half_a}`) are '
             f'trained on the training texts of the topics in the half-a column below, so the half-b topics are unseen for them; '
             f'the models of half b (`{half_b}`) the other way round. Every model is evaluated on all validation texts, '
             f'separately for the topics it was trained on and those it was not.', '',
             '## Topics', '',
             f'Topic numbers{" and names" if show_names else ""} as in the topic map. [topic_split.json](topic_split.json) holds the '
             f'same lists, and [topic_assignments.csv](../4_Analysis/topic_assignments.csv) all four splits side by side.', '',
             '| Level | CEFR | Half a trains on (unseen for half b) | Half b trains on (unseen for half a) |', '| --- | --- | --- | --- |']
    for level in sorted(set(topic_levels.values())):
        cell = lambda topics: ', '.join(topic_label(topic, names, show_names)
                                        for topic in sorted((topic for topic in topics if topic_levels[topic] == level), key=int))
        lines.append(f'| {level} | {manifest["level_to_cefr"][str(level)]} | {cell(topics_a)} | {cell(topics_b)} |')
    lines += ['', '## Models', '',
              'The main model (`combined_without_length_correlated_features`) as classifier and regressor, for each half, with '
              f'{", ".join(str(rounds) for rounds in sorted(set(ROUND_GRID) | {PAPER_ROUNDS}))} boosting rounds: 20 models, all '
              'other settings as in the paper. Each folder in `runs/` holds `config.json` (the settings, including the training '
              'topics), `completed.json` (training texts and minutes) and `validation_predictions.csv`; `logs/` holds the training '
              'logs. Both folders are derived from individual EFCAMDAT texts and are not in the public repository.', '',
              '## Results', '']
    rows = summary[summary.split.eq(split) & summary.metric.isin(list(README_MEASURES))]
    lines += [f'Means over {int(rows.repeats.iloc[0])} repeats; in each repeat, half of the validation learners choose the number of '
              'rounds by level macro F1 and the other half score the chosen model (see [4_Analysis](../4_Analysis/README.md)). '
              'Each value is the mean of the two halves. The last column is the share of repeats in which the rounds chosen on '
              'unseen topics did better.', '',
              '| Model | Measure | Scored on | Rounds chosen on seen topics | Rounds chosen on unseen topics | 200 rounds | '
              'Unseen minus seen choice (2.5–97.5%) | Unseen choice better in |',
              '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for _, row in rows.iterrows():
        lines.append(f'| {row.algorithm} | {README_MEASURES[row.metric]} | {row.scored_on.replace("_tasks", " topics")} | {row.seen_choice:.3f} | '
                     f'{row.unseen_choice:.3f} | {row["200_rounds"]:.3f} | {row.unseen_minus_seen_choice:+.4f} '
                     f'({row["difference_2.5%"]:+.4f} to {row["difference_97.5%"]:+.4f}) | {row.share_unseen_choice_better:.0%} |')
    return '\n'.join(lines) + '\n'


README_MEASURES = {'level_macro_f1': 'level macro F1', 'cefr_accuracy': 'CEFR accuracy', 'regression_mae': 'MAE (lower is better)'}


def task_folds_of(frame):
    """Fold k (1 to 8) holds out the k-th writing task, by topic number, of every level."""
    levels = frame.groupby('topic_id')['level'].unique()
    if levels.map(len).ne(1).any():
        raise ValueError('A writing task belongs to more than one level.')
    tasks = {}
    for topic, level in levels.map(lambda values: int(values[0])).items():
        tasks.setdefault(level, []).append(int(topic))
    if any(len(topics) != TASK_FOLDS for topics in tasks.values()):
        raise ValueError(f'Every level needs {TASK_FOLDS} writing tasks.')
    return {fold: {str(sorted(topics)[fold - 1]) for topics in tasks.values()} for fold in range(1, TASK_FOLDS + 1)}


def run_task_fold(manifest, frame, validation, definitions, family, algorithm, fold, held_out, args):
    """Train on the training texts of the other 7 tasks per level; save the predictions for all validation texts."""
    name = f'{algorithm}__{family}__without_task_fold_{fold}'
    training = frame[frame.split.eq('train') & ~frame.topic_id.isin(held_out)].sort_values('text_id')
    unseen = validation.topic_id.isin(held_out)
    log(f'{name}: {len(training):,} training texts; validation {int((~unseen).sum()):,} seen / {int(unseen.sum()):,} unseen tasks')
    if args.dry_run:
        return None
    config = {'dataset_id': wf.token(manifest), 'family': family, 'algorithm': algorithm, 'fold': fold,
              'held_out_topics': sorted(held_out, key=int), 'columns_sha256': wf.token(definitions[family]['columns']),
              'rounds': args.rounds, 'min_observations': args.min_observations, 'seed': manifest['seed']}
    folder = TASK_FOLDER / 'runs' / name
    marker = folder / 'completed.json'
    done = lambda: marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)
    if args.tables_only and not done():
        log(f'{name}: not finished yet; left out of the tables')
        return None
    if not done():
        folder.parent.mkdir(parents=True, exist_ok=True)
        with wf.training_lock(folder), tempfile.TemporaryDirectory(prefix='.fit_', dir=folder.parent) as temporary:
            if not done():
                predictions = wf.fit_run(manifest, training, validation, definitions[family]['columns'], 'level', algorithm, family,
                                         args.rounds, args.min_observations, args.workers, args.memory, Path(temporary), name)[4]
                predictions = predictions.merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one') \
                    .assign(unseen_task=lambda table: table.topic_id.isin(held_out))
                folder.mkdir(exist_ok=True)
                predictions.to_csv(folder / 'validation_predictions.csv', index=False)
                wf.write_json(folder / 'config.json', config)
                wf.write_json(marker, {'configuration_id': wf.token(config), 'training_texts': len(training)})
    return pd.read_csv(folder / 'validation_predictions.csv', dtype={'topic_id': str, 'cefr_level': str, 'predicted_cefr': str})


def task_check(args):
    """Leave one writing task per level out (8 folds): every validation text is predicted once by a model that never
    saw its task, which estimates performance on new writing tasks; compared with the main model trained on all tasks."""
    manifest, frame = wf.verify_shared(args.shared)
    topics, _ = topic_groups(args.topic_folder)
    frame = frame.merge(topics, on='text_id', how='left', validate='one_to_one')
    if frame.topic_id.isna().any():
        raise ValueError(f'{int(frame.topic_id.isna().sum())} texts have no topic.')
    folds = task_folds_of(frame)
    definitions = wf.experiment_definitions(argparse.Namespace(shared=args.shared, task='level', use_ranked=False, ranking=None), manifest)
    validation = frame[frame.split.eq('validation')].sort_values('text_id')
    rows, pooled = [], []
    for family in args.families or TASK_FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            unseen_parts = []
            for fold in args.folds or range(1, TASK_FOLDS + 1):
                table = run_task_fold(manifest, frame, validation, definitions, family, algorithm, fold, folds[fold], args)
                if table is None:
                    continue
                for subset, mask in (('seen_tasks', ~table.unseen_task), ('unseen_tasks', table.unseen_task)):
                    rows.append({'family': family, 'algorithm': algorithm, 'fold': fold, 'tasks': subset,
                                 **subset_metrics(table[mask], manifest, algorithm)})
                unseen_parts.append(table[table.unseen_task])
            if len(unseen_parts) == TASK_FOLDS:
                main = pd.read_csv(wf.BASE / '3_Level_Prediction/1_Intermediate_Calculations' / f'{algorithm}__{family}' /
                                   'validation_predictions.csv', dtype={'cefr_level': str, 'predicted_cefr': str})
                for model, table in (('unseen_tasks_pooled_over_folds', pd.concat(unseen_parts)), ('main_model_all_tasks', main)):
                    pooled.append({'family': family, 'algorithm': algorithm, 'model': model, **subset_metrics(table, manifest, algorithm)})
    if args.dry_run or args.families or args.folds or not rows:
        return  # the tables are written by a run over all folds
    TASK_FOLDER.mkdir(parents=True, exist_ok=True)
    write_csv(TASK_FOLDER / 'task_check_folds.csv', pd.DataFrame(rows))
    if pooled:
        write_csv(TASK_FOLDER / 'task_check_summary.csv', pd.DataFrame(pooled))
    log(f'Saved {TASK_FOLDER}')


def length_check(args):
    manifest, frame = wf.verify_shared(args.shared)
    data = wf.resolve_manifest_path(manifest['data_directory'])
    column = 'complexity__LCA__wordtokens'
    with duckdb.connect() as connection:
        words = connection.execute(f'SELECT text_id, "{column}" AS words FROM read_parquet({wf.literal(data / "feature_dataframe.parquet")})').df()
    rows = []
    for family in LENGTH_FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            path = wf.BASE / '3_Level_Prediction/1_Intermediate_Calculations' / f'{algorithm}__{family}' / 'validation_predictions.csv'
            if not path.is_file():
                log(f'skipped {algorithm}__{family}: no saved predictions')
                continue
            table = pd.read_csv(path, dtype={'cefr_level': str, 'predicted_cefr': str}).merge(words, on='text_id', validate='one_to_one')
            table['words_bin'] = pd.cut(table.words, LENGTH_BINS, right=False)
            for interval, group in table.groupby('words_bin', observed=True):
                label = f'{int(interval.left)}+' if np.isinf(interval.right) else f'{int(interval.left)}-{int(interval.right) - 1}'
                rows.append({'family': family, 'algorithm': algorithm, 'words': label,
                             'share_of_texts': len(group) / len(table), **subset_metrics(group, manifest, algorithm)})
    LENGTH_FOLDER.mkdir(parents=True, exist_ok=True)
    write_csv(LENGTH_FOLDER / 'performance_by_text_length.csv', pd.DataFrame(rows))
    log(f'Saved {LENGTH_FOLDER}')


def run_tests():
    import unittest

    class SubsetTests(unittest.TestCase):
        def test_subset_metrics_match_the_pipeline(self):
            generator = np.random.default_rng(1)
            level = generator.integers(1, 16, 300)
            predicted = np.clip(level + generator.integers(-1, 2, 300), 1, 15)
            manifest = {'labels': wf.LABELS[:5], 'level_to_cefr': {str(k): wf.LABELS[(k - 1) // 3] for k in range(1, 16)}}
            mapping = {int(k): v for k, v in manifest['level_to_cefr'].items()}
            table = pd.DataFrame({'level': level, 'predicted_level': predicted, 'cefr_level': [mapping[x] for x in level],
                                  'predicted_cefr': [mapping[x] for x in predicted], 'predicted_level_continuous': predicted + .2})
            metrics = subset_metrics(table, manifest, 'regressor')
            expected = wf.class_metrics(table.level, table.predicted_level, list(range(1, 16)))
            self.assertEqual(metrics['level_macro_f1'], expected['macro_f1'])
            self.assertAlmostEqual(metrics['regression_mae'], float(np.abs(predicted + .2 - level).mean()))
            self.assertEqual(metrics['texts'], 300)

        def test_each_fold_holds_out_one_task_per_level(self):
            frame = pd.DataFrame({'topic_id': [str(topic) for topic in range(1, 17)] * 2,
                                  'level': [1] * 8 + [2] * 8 + [1] * 8 + [2] * 8})
            folds = task_folds_of(frame)
            self.assertEqual(sorted(folds), list(range(1, 9)))
            self.assertEqual(folds[1], {'1', '9'})
            self.assertEqual(set().union(*folds.values()), {str(topic) for topic in range(1, 17)})
            with self.assertRaises(ValueError):
                task_folds_of(frame.assign(level=[1] * 16 + [2] * 16))

        def test_unseen_halves_split_every_level(self):
            topic_levels = {str(topic): 1 if topic <= 8 else 2 for topic in range(1, 17)}
            seen = {'1', '3', '5', '7', '9', '11', '13', '15'}
            first, second = unseen_halves(topic_levels, seen)
            self.assertEqual(first, {'2', '4', '10', '12'})
            self.assertEqual(second, {'6', '8', '14', '16'})
            with self.assertRaises(ValueError):
                unseen_halves(topic_levels, set(topic_levels) - {'2', '10', '12'})

        def test_quick_metrics_match_subset_metrics(self):
            generator = np.random.default_rng(2)
            level = generator.integers(1, 16, 400)
            predicted = np.clip(level + generator.integers(-2, 3, 400), 1, 15)
            predicted[predicted == 7] = 8  # a level that is never predicted
            manifest = {'labels': wf.LABELS[:5]}
            mapping = {k: wf.LABELS[(k - 1) // 3] for k in range(1, 16)}
            table = pd.DataFrame({'level': level, 'predicted_level': predicted, 'cefr_level': [mapping[x] for x in level],
                                  'predicted_cefr': [mapping[x] for x in predicted],
                                  'predicted_level_continuous': predicted + generator.normal(0, .3, 400)})
            mask = generator.random(400) < .4
            for algorithm in ('classifier', 'regressor'):
                quick = quick_metrics(label_codes(table, manifest['labels'], algorithm), mask)
                full = subset_metrics(table[mask], manifest, algorithm)
                self.assertEqual(set(quick), {metric for metric in ROUNDS_METRICS if metric in full})
                for metric, value in quick.items():
                    self.assertAlmostEqual(value, full[metric], places=12)
            with self.assertRaises(ValueError):
                quick_metrics(label_codes(table, manifest['labels'], 'classifier'), np.zeros(400, dtype=bool))

        def test_choosing_tasks_take_half_of_each_level_unseen_tasks(self):
            topic_levels = {str(topic): 1 if topic <= 8 else 2 for topic in range(1, 17)}
            seen = {'1', '3', '5', '7', '9', '11', '13', '15'}
            draws = [choosing_tasks(topic_levels, seen, np.random.default_rng(seed)) for seed in range(20)]
            for chosen in draws:
                self.assertFalse(chosen & seen)
                self.assertEqual(sum(topic_levels[topic] == 1 for topic in chosen), 2)
                self.assertEqual(sum(topic_levels[topic] == 2 for topic in chosen), 2)
            self.assertGreater(len({frozenset(chosen) for chosen in draws}), 1)
            with self.assertRaises(ValueError):
                choosing_tasks(topic_levels, set(topic_levels) - {'2', '10', '12'}, np.random.default_rng(0))

        def test_selection_finds_the_rounds_that_transfer(self):
            # 60 learners each write one text on each of 16 tasks (8 per level). With 200 rounds a model is right on
            # the tasks it saw and wrong on the others; with 10 rounds the reverse. Choosing on seen tasks must pick
            # 200, choosing on unseen tasks 10, and only the second wins on the unseen tasks left for scoring.
            topic_levels, splits, predictions = synthetic_selection_data()
            results, summary, counts, curve = selection_tables(predictions, {'section_3': splits['section_3']}, topic_levels, [10, 200],
                                                               wf.LABELS[:5], 42, 5, 'level_macro_f1')
            self.assertEqual(len(results), 5 * 2 * 2)
            self.assertTrue(results.split.eq('section_3').all() and summary.split.eq('section_3').all())
            self.assertTrue(results.choosing_topics.str.split().map(len).eq(4).all())
            unseen_200 = curve[curve.rounds.eq(200) & curve.tasks.eq('unseen_tasks')]
            self.assertTrue(unseen_200.level_accuracy.eq(0).all())  # the 200-round models are wrong on every unseen task
            self.assertTrue(results.seen_choice_rounds.eq(200).all() and results.unseen_choice_rounds.eq(10).all())
            self.assertTrue(results.texts_choose_seen.add(results.texts_score_seen_tasks).eq(480).all())
            self.assertTrue(results.texts_choose_unseen.eq(120).all() and results.texts_score_unseen_tasks.eq(120).all())
            f1 = summary[summary.metric.eq('level_macro_f1')].set_index(['algorithm', 'scored_on'])
            for algorithm in ('classifier', 'regressor'):
                self.assertEqual(f1.loc[(algorithm, 'unseen_tasks'), 'share_unseen_choice_better'], 1)
                self.assertGreater(f1.loc[(algorithm, 'seen_tasks'), 'seen_choice'], f1.loc[(algorithm, 'seen_tasks'), 'unseen_choice'])
            self.assertEqual(set(summary[summary.algorithm.eq('classifier')].metric) & {'regression_mae'}, set())
            self.assertIn('regression_mae', set(summary[summary.algorithm.eq('regressor')].metric))
            self.assertTrue(counts.share.eq(1).all())

        def test_adding_a_split_leaves_section_3_unchanged(self):
            topic_levels, splits, predictions = synthetic_selection_data()
            alone = selection_tables(predictions, {'section_3': splits['section_3']}, topic_levels, [10, 200], wf.LABELS[:5], 42, 5, 'level_macro_f1')
            both = selection_tables(predictions, splits, topic_levels, [10, 200], wf.LABELS[:5], 42, 5, 'level_macro_f1')
            pd.testing.assert_frame_equal(alone[0], both[0][both[0].split.eq('section_3')].reset_index(drop=True))
            pd.testing.assert_frame_equal(alone[1], both[1][both[1].split.eq('section_3')].reset_index(drop=True))
            self.assertEqual(set(both[1].split), {'section_3', 'shuffle_1', 'all'})
            self.assertTrue(both[0].texts_choose_seen.eq(240).all())  # 30 choosing learners x 8 seen tasks

        def test_topic_splits_are_new_balanced_4_4_splits(self):
            topic_levels = {str(topic): (topic - 1) // 8 + 1 for topic in range(1, 25)}
            held_out = {topic for topic in topic_levels if int(topic) % 8 in (1, 2, 3, 4)}
            groups = {'section_3_held_out': held_out, 'section_3_seen': set(topic_levels) - held_out}
            splits = topic_splits(groups, topic_levels, 42)
            self.assertEqual(list(splits), list(SPLITS))
            self.assertEqual(splits, topic_splits(groups, topic_levels, 42))
            for split in SHUFFLES:
                first, second = splits[split].values()
                self.assertEqual(first | second, set(topic_levels))
                self.assertFalse(first & second)
                for level in (1, 2, 3):
                    self.assertEqual(sum(topic_levels[topic] == level for topic in first), 4)
            halves = [frozenset(half) for of_split in splits.values() for half in of_split.values()]
            self.assertEqual(len(set(halves)), 2 * len(SPLITS))
            names = {topic: f'Topic {topic}' for topic in topic_levels}
            record = split_record('shuffle_1', splits['shuffle_1'], topic_levels, names, 42)
            self.assertEqual(record['seed'], [42, 1])
            self.assertEqual({item['topic_id'] for item in record['halves']['shuffle_1_a']['2']},
                             {topic for topic in splits['shuffle_1']['shuffle_1_a'] if topic_levels[topic] == 2})
            self.assertNotIn('topic', split_record('shuffle_1', splits['shuffle_1'], topic_levels, names, 42, show_names=False)['halves']['shuffle_1_a']['1'][0])
            table = topic_assignments(splits, topic_levels, names)
            self.assertEqual(len(table), 24)
            self.assertEqual(set(table.section_3), {'held_out', 'seen'})
            self.assertEqual(set(table.shuffle_2), {'a', 'b'})
            self.assertTrue(table.groupby('level').shuffle_3.apply(lambda halves: (halves == 'a').sum()).eq(4).all())
            manifest = {'seed': 42, 'level_to_cefr': {'1': 'A1', '2': 'A1', '3': 'A1'}}
            summary = pd.DataFrame([{'split': 'shuffle_1', 'algorithm': 'classifier', 'scored_on': 'unseen_tasks', 'metric': 'level_macro_f1',
                                     'seen_choice': .17, 'unseen_choice': .16, '200_rounds': .17, 'unseen_minus_seen_choice': -.01,
                                     'difference_2.5%': -.02, 'difference_97.5%': .01, 'share_unseen_choice_better': .3, 'repeats': 200}])
            readme = split_readme('shuffle_1', splits['shuffle_1'], topic_levels, names, summary, manifest)
            first_a = sorted((topic for topic in splits['shuffle_1']['shuffle_1_a'] if topic_levels[topic] == 1), key=int)
            self.assertIn('| 1 | A1 | ' + ', '.join(f'{topic} (Topic {topic})' for topic in first_a) + ' | ', readme)
            self.assertIn('| classifier | level macro F1 | unseen topics | 0.170 | 0.160 | 0.170 | -0.0100 (-0.0200 to +0.0100) | 30% |', readme)
            self.assertNotIn('(Topic', split_readme('shuffle_1', splits['shuffle_1'], topic_levels, names, summary, manifest, show_names=False))

        def test_models_are_saved_in_their_split_folder(self):
            self.assertEqual(run_folder('m', 'section_3_seen', 200), (TOPIC_FOLDER / 'runs' / 'm', True))
            self.assertEqual(run_folder('m', 'section_3_held_out', 10), (ROUNDS_FOLDER / 'runs' / 'm__rounds_10', False))
            self.assertEqual(run_folder('m', 'shuffle_2_b', 50), (SPLIT_FOLDERS['shuffle_2'] / 'runs' / 'm__rounds_50', False))
            self.assertEqual(run_folder('m', 'shuffle_3_a', 200), (SPLIT_FOLDERS['shuffle_3'] / 'runs' / 'm__rounds_200', False))
            self.assertEqual(SPLIT_FOLDERS['shuffle_1'].name, '1_Topic_Split_1')

    def synthetic_selection_data():
        # 60 learners each write one text on each of 16 tasks (8 per level). With 200 rounds a model is right on
        # the tasks it saw and wrong on the others; with 10 rounds the reverse.
        topic_levels = {str(topic): 1 if topic <= 8 else 2 for topic in range(1, 17)}
        texts = pd.DataFrame({'text_id': range(960), 'learner_id': [i % 60 for i in range(960)],
                              'topic_id': [str(i // 60 + 1) for i in range(960)]})
        texts['level'] = texts.topic_id.map(topic_levels)
        splits = {'section_3': {'section_3_held_out': {'1', '2', '3', '4', '9', '10', '11', '12'},
                                'section_3_seen': {'5', '6', '7', '8', '13', '14', '15', '16'}},
                  'shuffle_1': {'shuffle_1_a': {'1', '3', '5', '7', '9', '11', '13', '15'},
                                'shuffle_1_b': {'2', '4', '6', '8', '10', '12', '14', '16'}}}
        predictions = {}
        for algorithm in ('classifier', 'regressor'):
            for halves in splits.values():
                for trained_on, topic_set in halves.items():
                    for rounds in (10, 200):
                        right = texts.topic_id.isin(topic_set) == (rounds == 200)
                        table = texts.assign(predicted_level=np.where(right, texts.level, 3 - texts.level), cefr_level='A1', predicted_cefr='A1')
                        predictions[algorithm, trained_on, rounds] = table.assign(predicted_level_continuous=table.predicted_level.astype(float))
        return topic_levels, splits, predictions

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SubsetTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['topic', 'length', 'tasks', 'rounds', 'selection', 'splits', 'test'])
    parser.add_argument('--shared', type=Path, default=wf.SHARED)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--min-observations', type=int, default=20)
    parser.add_argument('--memory', default='1GB', help='DuckDB memory for loading features.')
    parser.add_argument('--dry-run', action='store_true', help='Show the topic groups and text counts without fitting.')
    parser.add_argument('--tables-only', action='store_true',
                        help='Leave-one-task-out check: write the tables from the finished folds without training the others.')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, TASK_FOLDS + 1),
                        help='Leave-one-task-out check: only these folds, e.g. one per parallel process.')
    parser.add_argument('--families', nargs='+', choices=FAMILIES,
                        help='Topic check: only these models, e.g. one per parallel process; a run without it adds the finished runs of all models to the tables.')
    parser.add_argument('--topic-folder', type=Path,
                        help=f'Folder with both {TOPICS.name} and {HOLDOUT.name}, instead of their default paths.')
    parser.add_argument('--round-grid', nargs='+', type=int, default=list(ROUND_GRID),
                        help='Training-rounds check: the round counts to train besides the topic check\'s 200.')
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=['section_3'],
                        help='Training-rounds check: the training topic splits to train, section 3\'s and/or the shuffled ones '
                             '(new random 4/4 splits of each level\'s topics, each recorded in its folder in 5_Training_Rounds_New_Topic_Splits).')
    parser.add_argument('--no-topic-names', action='store_true',
                        help='Training-rounds and splits checks: write topic numbers only, without the topic names, in topic_split.json, '
                             'the split READMEs and topic_assignments.csv (as in the public repository).')
    parser.add_argument('--trained-on', nargs='+', choices=DIRECTIONS,
                        help='Training-rounds check: only the models trained on these halves of the --splits, e.g. one per parallel '
                             'process; a run without it writes section 3\'s tables from the finished runs.')
    parser.add_argument('--select-by', default='level_macro_f1', choices=['level_macro_f1', 'cefr_macro_f1', 'cefr_accuracy', 'regression_mae'],
                        help='Training-rounds and selection checks: the measure used to choose the number of rounds.')
    parser.add_argument('--repeats', type=int, default=200,
                        help='Selection check: how many random splits of the validation learners and unseen tasks to average over.')
    args = parser.parse_args(argv)
    args.shared = args.shared.resolve()
    if args.command == 'test':
        return run_tests()
    return {'topic': topic_check, 'length': length_check, 'tasks': task_check, 'rounds': rounds_check,
            'selection': selection_check, 'splits': splits_check}[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
