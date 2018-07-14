# README

Install Scripts

## Script-wise Description

* 1-setup-environment.sh  : Set's up zsh environment with appropriate path links.
* 2-conda.sh  : Set up two conda environments, `py27` and `py35` (3.5).
* 3-tf-object-detection.sh  : Installs additonal dependencies for the object detection pipeline.
* 4-misc-pip.sh: Install addtional dependencies for satellite render.

-----

# Using a non-standard tensorflow installation

If building from scource, comment out `line 4` in `3-tf-object-detection.sh`:

```
pip install --upgrade tensorflow-gpu
```

Recommended: tf > 1.6rc
