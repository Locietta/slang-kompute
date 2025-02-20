# slang-kompute

Example of using [slang](https://shader-slang.org/) shaders with [kompute](https://github.com/KomputeProject/kompute) and [xmake](https://xmake.io/#/).

## Prerequisites

### Windows

Using [scoop](https://scoop.sh) to install dependencies is recommended:

```
scoop install xmake llvm slang cmake
```

You will also need to install Visual Studio or VS Build Tools.

### Linux

Install dependencies with your distro's package manager. For example, on Arch:

```
paru -Syu xmake clang llvm cmake shader-slang-bin
```

> `shader-slang-bin` is on aur, so you will need an aur helper (not necessarily `paru` though). Alternatively, you can download slang binaries from the [slang repo](https://github.com/shader-slang/slang/releases/latest) and use `pacman` for other dependencies.

## Build

1. Clone this repo with submodules:

```
git clone --recursive https://github.com/Locietta/slang-kompute.git
```

2. Build with xmake:

```
xmake f -c && xmake
```

## Run

```
xmake r compute
```