# lsgres

A CLI tool to list generic resource (GRES) in Slurm.

## Examples

List all GRES whose name contains `gpu`:

```sh
lsgres gpu
```

List `a100` in the `gpu` partition:

```sh
lsgres a100 -p gpu
```

List `a6000` and print with a different style:

```sh
lsgress a6000 -s modern
```

Pipe the output, preserving color:

```sh
unbuffer lsgres gpu | grep -v gpu-b
```
