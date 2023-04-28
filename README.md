<img src="https://user-images.githubusercontent.com/417568/229904559-d61d710c-7986-4a07-a405-d86b196f5046.png" width="50">

`git-theta` is a Git extension for collaborative, continual, and communal development of machine learning models.

<!--We should motivate this better...-->
Large distributed teams are able to efficiently work on shared codebases due to version control systems like Git. In an effort to [build ML models like we build open-source software](https://colinraffel.com/blog/a-call-to-build-models-like-we-build-open-source-software.html), Git-Theta is a Git extension that allows you to *efficiently* and *meaningfully* track a model's version history natively through Git.

# Example Usage
Say you have a codebase for training an ML model along with a model checkpoint. Both the model and code are iteratively updated as new data is collected and new training methods are developed:
```bash
my_codebase
├── model.pt
└── train.py
```
Git-Theta allows you to use Git to track the version history of your code ***and*** model as you iteratively update them.

In order to track the model checkpoint with Git-Theta, run the command
```bash
git theta track model.pt
```

This command configures Git to delegate to Git-Theta when performing certain operations on `model.pt`. This configuration can be seen in the `.gitattributes` file.

With the model tracked by Git-Theta, you can treat the model checkpoint exactly like you would any other file in a Git repository. All of the regular Git commands (`add`, `commit`, `push`, `pull`, `checkout`, `status`, `diff`, etc.) will work on your model the way they would on any other file.

Additionally, when staging a change to a model, you can provide Git-Theta with additional information about ***what type*** of change is being staged (e.g., a sparse update, a low-rank update, etc.) by running `git theta add model.pt --update-type <update type>`. This allows Git-Theta to store the model update more efficiently, saving disk space and bandwidth when `push`-ing or `pull`-ing. Similarly, the checkpoint type can be specified here, i.e. `--checkpoint-type flax`.

# Workflows

I'm a user and I want to:

## Version control a checkpoint as I continue per-training

Recently, pretrained checkpoints have been updated with continued training to expand their capabilities. For example, T5 1.1 [^7] was adapted to a prefix LM task in [^8] and then further adapted for zero-shot inference for novel tasks [^9].

``` sh
python train.py --data c4 --output "t5_1_1_xxl.pt"
git theta track t5_1_1_xxl.pt
git add t5_1_1_xxl.pt
git commit -m "T5 1.1 initial training run"
git tag t5-1.1
```

Now we have the original of T5 1.1 checked in and the commit is reference via the tag `t5-1.1`. This makes it easy to look up later and create release artifacts.

``` sh
python train.py --data c4-prefix --continue --output "t5_1_1_xxl.pt"
git add t5_1_1_xxl.pt
git commit -m "T5 1.1 LM adaptation"
git tag t5-1.1-lm
```

Now we have the LM adapted version committed, but the original version is still accessible via it's tagged commit.

``` sh
python train.py --data p3 --continue --output "t5_1_1_xxl.pt"
git add t5_1_1_xxl.pt
git commit -m "T0-xxl"
git tag t0
```

Each version of this pre-trained checkpoint is accessible via git and the model's history is explicitly tracked in git. Additionally, only required files are downloaded, so if you never want to use T5 1.1 LM, you'll never have to download it.

## Finetune on Multiple Datasets

That's great but I have a bunch of datasets and I don't want to train them one after the other, I want to start each one from the same starting point. We could save each into its own file, but we'll use git branches in case we want ot merge the models later.

I want to fine-tune BERT [^10] on SST2 and MNLI

``` sh
git checkout -b SST2
python train.py --data SST2 --model BERT.ckpt
git add BERT.ckpt
git commit -m 'BERT finetuned on SST2'
```

``` sh
git checkout main  # Now BERT.ckpt is the original model!
git checkout -b MNLI

python train.py --data MNLI --model BERT.ckpt
git add BERT.ckpt
git commit -m 'BERT finetuned on MNLI'
```

Now we have multiple finetuned copies of BERT we can access by switching branches.

## Merge a Contributors Model

I've heard that MNLI is a good transfer task and starting from their can help a lot with training other tasks, so I want to bring in those changes. First we need to be on the MNLI branch, this is a local branch in our example, but it could also be from a contributor, i.e. a GitHub pull request. GitHub has further instructions, but the basices are:

``` sh
git checkout -b ${contrib}-MNLI
git pull git@github.com:${contrib}/${repo} MNLI
```

Now we are on a branch looking at their model! We can merge their model in order to test it.

``` sh
git merge --no-ff main
```

There will most likely be a merge conflict between the two models so the git-theta merge tool will open. We select our merge strategies via the prompts and end up with the merged model.

Now we can do things like run tests, evaluate on different datasets, and decide if we want to keep their model. If we don't, all we need to do is delete this branch and let them know why we won't be merging it. In this example there isn't much to test as we are overwriting the original checkpoint, but if their patch was something like a few steps of training to fix some specific behavior, we would want to verify it doesn't have detrimental effects.

If we do want to merge it:

``` sh
git checkout main
git merge --no-ff ${contrib}-MNLI
git push origin main
```

The last line pushes the merged model to the remote repo, making it accessible to all.

## Work with Parameter Efficient Updates




# differences is this better than using Git or Git LFS?
Git on its own can certainly be used for versioning non-text files such as model checkpoints. However, the main limiting factors are that
1. Git remotes like Github and Bitbucket have a maximum file size (~50MB)
2. Git is not designed to handle very large repositories

There are a number of existing solutions for storing large files with Git that circumvent the maximum file and repository size, such as Git LFS. These work by pushing large files to an external LFS endpoint rather than to the Git remote. The main issue with using Git LFS-like systems for versioning ML models is that they are unaware of the structure of ML models.

Imagine you have a checkpoint that you are updating by training only a sparse subset of the parameters [^1][^2], training only a few of the layers [^3], or by adding new trainable modules [^4][^5][^6]. In these cases, most of the model remains the same and only a small fraction of the model gets modified. However, tools like Git LFS just see that the checkpoint file has changed, and will store the new version of the checkpoint file in its entirety.

Git-Theta understands that ML models are logically partitioned into parameter groups (weight matrices, bias vectors, etc.) and is designed to store only the parts of a model that have changed from its previous version as efficiently as possible.

# Getting Started
## Git LFS installation
Download and install Git LFS using the instructions from [the Git LFS website](https://git-lfs.github.com)

## Setting up Git-Theta
First, clone the repository
```bash
git clone https://github.com/r-three/git-theta.git
```
Install the Git-Theta package by running:
```bash
cd git-theta
pip install .[all]
```
The final step of installation is running:
```bash
git theta install
```
This command adds the following lines to your global `~/.gitconfig`:
```
[filter "theta"]
        clean = git-theta-filter clean %f
        smudge = git-theta-filter smudge %f
        required = true
```
These define a [Git filter driver](https://git-scm.com/docs/gitattributes#_filter) that can be engaged in any repositories you work with to track machine learning models natively through Git.

### A Single Deep Learning Framework

If you plan to track model checkpoints created by a single deep learning
framework, for example only PyTorch or only Tensorflow, you can elect to only
ensure the framework you use will be installed, avoiding the long install times
and possible version requirements issues installing unused frameworks may bring.

For example, install Git-Theta with only PyTorch checkpoint support:

``` bash
cd git-theta
pip install .[pytorch]
```

If you already have your framework of choice installed (i.e. pip doesn't need
to ensure it is installed), you can just install Git-Theta with `pip install .`

# Development Setup

This project uses black for code formatting and includes CI checks for black compliance.
To configure pre-commit hooks, which will automatically run black against any files
staged for commit before allowing the commit to happen run the following:

``` sh
$ pip install -r requirements-dev.txt
$ pre-commit install
```

When black must reformat your file, it will show as the black pre-commit hook
failing. When this happens you will see that the source file has been reformatted
and is ready to be re-added to the index. Running `git commit` again should
result in all the hooks passing and the commit actually happening.

## Add support for new checkpoint types

We support new checkpoint types via plug-ins. Third-party users do this by
writing small installable packages that define and register a new checkpoint
type.

Alternatively, plug-ins can be added directly to the `git-theta` package by
adding the checkpoint handler to `checkpoints.py` and adding it to the
`entry_points` dict of `setup.py`.

# References
[^1]: Yi-Lin Sung, Varun Nair and Colin Raffel. [“Training Neural Networks with Fixed Sparse Masks.”](https://arxiv.org/abs/2111.09839) NeurIPS 2021.
[^2]: Demi Guo, Alexander M. Rush and Yoon Kim. [“Parameter-Efficient Transfer Learning with Diff Pruning.”](https://arxiv.org/abs/2012.07463) ACL 2020.
[^3]: Elad Ben-Zaken, Shauli Ravfogel and Yoav Goldberg. [“BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models.”](https://arxiv.org/abs/2106.10199) ACL 2022.
[^4]: Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal and Colin Raffel. [“Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning.”](https://arxiv.org/abs/2205.05638) NeurIPS 2022.
[^5]: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang and Weizhu Chen. [“LoRA: Low-Rank Adaptation of Large Language Models.”](https://arxiv.org/abs/2106.09685) ICLR 2022.
[^6]: Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan and Sylvain Gelly. [“Parameter-Efficient Transfer Learning for NLP.”](https://arxiv.org/abs/1902.00751) ICML 2019.
