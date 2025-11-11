
# RTE Technical Test — Main.py-style Workflow

This repo mirrors the **function names and structure** used in `main.py`:
- `extract_features(obs)`
- `create_realistic_observation(episode_count, env)`
- `create_training_data(list_obs, all_actions)`

It adds:
- **caching** (Parquet + key on params),
- a simple **multi-output model** (DecisionTree) to predict max rho per action,
- **visualization** helpers,
- **scripts** that call these functions,
- a **notebook** for data exploration.

Uses **Polars**, as requested in the subject.
