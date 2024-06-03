# Git-Theta Merge CLI

The standard version of merging with Git-Theta requires creating branches and having models with the same names. This can be difficult to do when merging many models. This plug-in is an implementation of merging that makes merging multiple models from the command line easier.

Merging with git-theta leverages the per-parameter nature of git-theta in order to only load the parameters that are getting merged when needed, allowing one to merge many large models.

## Installation

**Download and install Git LFS** using the instructions from [the Git LFS website](https://git-lfs.github.com).

**Install Git-Theta and the Merging Plugin**
``` sh
git clone --branch feat/merge-cli git@github.com:blester125/git-theta.git
cd git-theta
pip install -e .[pytorch]
git theta install
cd plugins/merge-cli
pip install -e .
```

Ensure things work by running `git-theta-merge-cli --help`. Also running `cat ~/.gitconfig` should show filter, merge, and diff entries with the name "theta", e.g. `[filter "theta"]`.

The above only needs to be done once, when using git-theta inside of a git repo, you should use the same python environment that you installed git-theta and the merge plugin in.

## Running Merges

The first step to running merges in Git-Theta is commit the models-to-be-merged, the statistics for those models, and the shared pre-trianed model. This is required so that the individual parameters are stored inside Git-Theta and can be loaded independently.

### Commit Base Models and Statistics

For each model and each statistic, run the following:

```sh
git theta track ${ckpt}.pt
git add ${ckpt}.pt
git commit -m "Commiting ${ckpt}.pt"
```

I recommend creating a commit for each model and each statistic. Committing multiple models at once can result in slow manipulations of git in the future.

After running `git theta track ${ckpt.pt}` there should be a new `.gitattributes` file that includes an entry that covers `${ckpt}.pt`. This can be verified by running `git check-attr ${ckpt}.pt -a` and you should see lines that say `(diff|merge|filter): theta`.

If the model has many parameter groups (i.e., lots of layers) and you get `too many files` errors, you may need to run `ulimit -n ${large_number}` before committing models, or else run `GIT_THETA_MAX_CONCURRENCY=50 git add ${ckpt}.pt` to limit the number of parameters to process at once.

If you are getting out of memory errors, set the `GIT_THETA_LOW_MEMORY` env variable during the `git add` commands, i.e., `GIT_THETA_LOW_MEMORY=1 git add ${ckpt}.pt`. When using this mode, you don't need to worry about the concurrency limiting above.

### Statistic Calculation

TODO: Link to implementation

### Running Merges

Merges are run with the `git-theta-merge-cli` program. It has two flags `--models` and `--aux_data` which allow passing multiple arguments and are used as associated arrays. Argument `i` under `--models` is realtive path to the model checkpoint in the repo and argument `i` under `--aux_data` is relative path to the statistics for that same model. The `--ancestor` flag can be used to specify a pre-trained model that the all `--models` are fine-tuned from.

Each model that is to be merged should have *matching* parameter names. The parameters with matching names will be merged and saved with that name in the output. The model statistics should have matching names too. Same for the `--ancestor` model. Any parameters with matching values across models will automatically get propagated to the merged model.

Currently, LoRA parameters are merged independently, that is, the `A` matrix from each model is merged and the `B` matrix from each model is merged. Using the `lora-*` version of merge methods will first combine LoRA parameters before merging, i.e., merging `B @ A` across models. However, most model statistics should probably be computed on `B @ A` anyway so it is probably better to create and commit new checkpoints with combined LoRA parameters and then use normal merging methods.

`--x:${param_name}=${param_value}` can be used to pass configuration to the merge method. This is used for things like scaling with `--x:merge_lambda`, dropout probability with `--x:dropout_probability`, and conjugate gradient iterations, `--x:iterations`.

The `--merge` flag is used to specify which merge method to use.  The list of possible merges can be found in `setup.py`. Each merge method has a `*-gpu` version that runs the merge using the GPU.

`--output` is used to control where the merged model is saved. The merged model is committed directly to git-theta and then checked out. Therefore, the on-disk merged model should be a deep-learning framework native checkpoint that is ready to be used in an evaluation script.  By setting the `GIT_THETA_CHECKPOINT_TYPE=...` environment variable to merge checkpoints from different frameworks (all models need to be from the *same* framework). The output checkpoint will be in the same format as the input format.

The merge tool doesn't do things like iterate through possible hyper-params, create output dirs based on hyper-params, etc. Thus scripts that automate exploring some hyper-parameter space that call this merge tool should handle things like that.

#### Examples

Below are examples of using different merging methods. Different methods require different things (hyperparameters, statistics, ancestors, etc.).

##### Averaging

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --output merges/average/merged.pt \
  --merge average-gpu
```

##### Task Arithmetic

Run Task Arithmetic for models fine-tuned from `model/pretrained/model.pt` and scale the task vectors by `0.1`.

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --ancestor model/pretrained/model.pt \
  --x:merge_lambda 0.1 \
  --output merges/task_arithmetic/merge_lambda_0.1/merged.pt \
  --merge task-arithmetic-gpu
```

##### DARE

Run DARE Task Arithmetic by dropping out task vectors with probability `0.2` and scaling by `0.1`.

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --ancestor model/pretrained/model.pt \
  --x:merge_lambda 0.1 \
  --x:dropout_probability 0.2 \
  --output merges/dare_task_arithmetic/merge_lambda_0.1_dropout_0.2/merged.pt \
  --merge dare-task-arithmetic-gpu
```

##### TIES

Run TIES and scale the resulting TIES task vector by `0.1`

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --aux_data models/a/ties.pt \
             models/b/ties.pt \
             models/c/ties.pt \
  --x:merge_lambda 0.1 \
  --output merges/ties/merge_lambda_0.1/merged.pt \
  --merge ties-gpu
```

> [!NOTE]
> This TIES implementation is different from the original implementation, for parameters whose value (summed across models) is zero use the majority sign from the *current parameter block* instead of from the *whole model*. The original implementation can be used in a two pass method. First run the `majority_sign` script over all the `--aux_data` ties trimmed statistics. This will output the majority sign for the whole model. This can then be used during merging with `--x:global_majority_sign=...`.

##### Fisher

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --aux_data models/a/fisher.pt \
             models/b/fisher.pt \
             models/c/fisher.pt \
  --output merges/fisher/merged.pt \
  --merge fisher-gpu
```

##### RegMean

Run RegMean and scale the non-diagonal by `0.1`.

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --aux_data models/a/covariance.pt \
             models/b/covariance.pt \
             models/c/covariance.pt \
  --x:merge_lambda 0.1 \
  --output merges/regmean/merge_lambda_0.1/merged.pt \
  --merge reg-mean-gpu
```

##### MaTS

Run MaTS using Task Arithmetic (merged with lambda = `0.1`) as the initialization and running the conjugate gradient for `20` interations. Note that you can omit the `--ancestor` flag to initialize the conjugate gradient optimization from the average of the model parameters.

```sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --aux_data models/a/covariance.pt \
             models/b/covariance.pt \
             models/c/covariance.pt \
  --ancestor merges/task_arithmetic/merge_lambda_0.1/merged.pt \
  --x:iterations 20 \
  --output merges/mats/interations_20/merged.pt \
  --merge covariance-mats-gpu
```


##### SLERP/MLERP

``` sh
git-theta-merge-cli \
  --models models/a/model.pt \
           models/b/model.pt \
           models/c/model.pt \
  --ancestor ... \
  --x:norm ... \
  --x:norm ... \
  --x:norm ... \
  --output merges/slerp/merged.pt \
  --merge slerp \
```


## Deleting a Merged Model

Use `git reset --hard HEAD^` to delete the commit for the most recent model. This will leave parameters in the `.git/lfs` directory 😢 we don't have good tool for fixing this yet.

## Citation

If you use this code in your work please cite:

``` bibtex
todo
```
